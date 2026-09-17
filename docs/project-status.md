# 静态权重缓存插件 —— 现状与验证报告

> 日期：2026-09-17
> 环境：openEuler aarch64 / Ascend 910 / 单容器单机
> 结论速览：**功能链路已打通并通过逐字节正确性验证；在当前"单机 + page cache 命中"场景下性能为 0.90x（无收益）**，原因与真实收益场景见第 4.3 节。

---

## 1. NFS 共享盘压测

### 1.1 环境

| 项 | 值 |
| --- | --- |
| 挂载点 | `172.26.2.10:/Common/y50063564` → `/data/y50063564` |
| 协议 | NFSv3, `rsize=1048576`(1MB), `wsize=1048576`(1MB), `proto=tcp`, `hard`, `timeo=600` |
| 快网卡 | `enp162s0f0` / `172.26.4.186` / **100 Gb/s** / driver `hisdk3` |
| 慢网卡 | `enp48s3u1u1` / `80.48.17.186` / **1 Gb/s** / driver `ax88179_178a`（USB 3.0 千兆网卡） |
| 默认路由 | **走慢网卡** —— 所以 `socket.connect(("8.8.8.8",80))` 得到的是 `80.48.17.186` |
| NFS 实际路径 | 经快网卡到达（`connect(("172.26.2.10",2049))` → `172.26.4.186`） |
| 测试文件 | `qwen3-8B` 的 5 个 safetensors，合计 16,381,516,776 B = **15.26 GiB** |

### 1.2 操作指令（命令行）

压测的关键是**绕过 page cache**。本机 2 TB 内存，读一遍 15 GiB 的模型就全部进了 page cache，之后量到的是本地内存带宽而不是 NFS。用 `O_DIRECT` 打开文件即可：page cache 既不参与服务也不吸收数据，每次读都会真实地发一次 NFS READ，且**不需要**动全局 page cache（`drop_caches` 会影响同机其他用户）。

**单流基线：**

```bash
cd /data/y50063564/qwen3-8B-dfly/qwen3-8B
dd if=model-00001-of-00005.safetensors of=/dev/null bs=1M count=1024 iflag=direct
```

**多流并发（把同一条命令并发起来）：**

```bash
F=model-00001-of-00005.safetensors
for n in 1 2 4; do
  S=$(date +%s%N)
  for i in $(seq $n); do
    dd if=$F of=/dev/null bs=1M count=2048 iflag=direct 2>/dev/null &
  done
  wait
  E=$(date +%s%N)
  awk -v n=$n -v s=$S -v e=$E 'BEGIN{t=(e-s)/1e9; g=n*2;
       printf "streams=%d  %dGiB in %.2fs  => %.2f GiB/s  %.2f Gbps\n", n, g, t, g/t, g*8/t}'
done
```

**其他常用指令：**

```bash
# 不带 O_DIRECT —— 量的是 page cache 命中时的内存带宽，不是 NFS
dd if=$F of=/dev/null bs=1M

# 整文件、看总体平均
dd if=$F of=/dev/null bs=4M iflag=direct status=progress

# 对比：写方向
dd if=/dev/zero of=/data/y50063564/scratch.bin bs=4M count=1024 oflag=direct

# 观测 RPC 计数（判断有没有真实流量；静置 5s 数值不变说明没有 NFS 活动）
grep -E "^rpc|^proc3" /proc/net/rpc/nfs

# 客户端 RPC 挂起重传统计
nfsstat -c
```

### 1.3 结果

用 `bench/nfs_scan.py`（O_DIRECT + mmap 缓冲区，4 MiB 分块）扫流数，每点跑 2 遍：

| 流数 | Gbps | GiB/s | 墙钟 | p50 时延 | p99 时延 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8.92 | 1.03 | 29.40s | 3.7 ms | 8.4 ms |
| **2** | **13.68** | **1.59** | **19.15s** | 4.4 ms | 15.6 ms |
| 4 | 13.05 | 1.52 | 20.09s | 10.0 ms | 24.2 ms |
| 6 | 13.03 | 1.52 | 20.12s | 14.9 ms | 29.1 ms |
| 8 | 12.78 | 1.49 | 20.50s | 20.0 ms | 43.9 ms |
| 12 | 12.59 | 1.46 | 20.82s | 30.7 ms | 60.9 ms |
| 16 | 12.61 | 1.47 | 20.78s | 41.1 ms | 91.7 ms |
| 24 | 12.29 | 1.43 | 21.33s | 64.7 ms | 87.7 ms |
| 32 | 11.64 | 1.36 | 22.51s | 91.3 ms | 112.0 ms |

用纯 `dd` 命令复现（1.2 节指令，每流 2 GiB）：

| 流数 | GiB/s | Gbps |
| ---: | ---: | ---: |
| 1 | 0.79 | 6.31 |
| 2 | 1.27 | 10.17 |
| 4 | 1.57 | 12.53 |

两条路径结论一致：**上限约 1.6 GiB/s（13 Gbps）**，与用哪个工具无关。

### 1.4 分析

**拐点在 2 条流，不是 12 条。** 2 流即达峰，之后吞吐**不升反降**（13.68 → 11.64 Gbps），而 p50 时延随流数**线性上涨**（3.7 ms → 91.3 ms）。这是典型的"并发额度被卡死"特征：不是服务端在扩容，而是请求在客户端排队。

**根因已定位到客户端挂载配置：**

```
/proc/sys/sunrpc/tcp_slot_table_entries = 2
```

挂载参数里**没有 `nconnect`**（默认单条 TCP 连接）。合起来就是：

> 整条链路上同时最多只有 **2 个在飞的 RPC**，每个 1 MiB（rsize）→ 在途仅 2 MiB。

开再多线程，也只是让请求在客户端排队，所以时延线性涨、吞吐不动。

**瓶颈不在硬件：**

- 网卡是 100 GbE，13.68 Gbps 只用到链路的 **13.7%**
- `tcp_max_slot_table_entries = 65536`，说明这个值**可以放开**
- 服务端 IP（`172.26.2.10`）与客户端快网卡同网段，路径不经过慢网卡

**与预期不符**：曾预期"不到 12 流达到 70-80 Gbps"，实测上限 13.68 Gbps，差约 5 倍。这很可能是因为那条经验来自另一套挂载配置（`nconnect` 更大或 slot table 已调优）。

### 1.5 对项目的含义

按当前配置，15.26 GiB 模型的冷加载时间下限：

| 场景 | 耗时 |
| --- | --- |
| 单流 | ≈ 15 s |
| 当前最优（2 流） | ≈ 10 s |
| 实测（服务端单线程加载） | **14 s**（等效 1.17 GB/s） |

**不改挂载配置的话，冷路径再怎么优化也压不到 10 秒以下** —— 这就是优化空间的边界。若能放开 slot table 并接近 70-80 Gbps，冷加载会降到 **~2 秒**级别，那时插件的价值定位需要重新评估。

---

## 2. 框架功能架构

### 2.1 角色与部署形态

```
┌──────────────────────── 缓存服务进程 (长驻) ────────────────────────┐
│  WeightCacheServer                                                 │
│   ├── HostMemManager      一整块 host 内存池（线性分配）             │
│   ├── ModelManager        safetensors → 内存池，产出元数据           │
│   └── TransferBackend     数据面（tcp / mooncake）                  │
│                                                                    │
│  ┌── 控制面: ZMQ REP  tcp://ip:port ──┐  ┌── 数据面: TCP ip:port ──┐ │
│  │  只回元数据，不回权重数据          │  │  按 (addr,len) 回字节   │ │
│  └────────────────────────────────────┘  └─────────────────────────┘ │
└────────────────────────────────────────────────────────────────────┘
                    ▲ ①查元数据              ▲ ②按地址读 bucket
                    │  (ZMQ REQ/REP)         │  (TCP)
┌───────────────────┴────────────────────────┴───────────────────────┐
│  StaticWeightCacheClient  (跑在 vLLM 进程侧)                        │
│    recv_buf  本地接收缓冲区（已注册）                                │
│    receive_weights()  →  async generator of (name, Tensor)          │
└────────────────────────────────────────────────────────────────────┘
```

核心思路：**权重只从共享盘读一次，驻留在服务进程的 host 内存里；之后所有 vLLM 进程改为向这个服务进程要数据。**

服务进程常驻，vLLM 重启不影响缓存 —— 这正是"避免反复从共享盘拉取"的关键。

### 2.2 组件清单

| 文件 | 职责 |
| --- | --- |
| `host_mem_manager.py` | 一整块长驻 host 内存池 + 线性 bump 分配器；4 KiB 对齐；pinned 可选；`mlock` 可选 |
| `model_manager.py` | safetensors → bucket 打包；产出 `ModelMetadata` |
| `protocol.py` | 控制面线格式；`to_weight_info()` 对齐 verl-recipe flexfetch 的结构 |
| `server.py` | 常驻服务进程：内存池 + 元数据服务 + 数据面 |
| `client.py` | 拉取侧：`receive_weights()` 异步生成器 |
| `transfer_backend.py` | 数据面抽象 + `TcpTransferBackend` / `MooncakeTransferBackend` |
| `checkpoint_engine.py` | 包成 verl `CheckpointEngine` 形状，注册名 `static_weight_cache`，只支持接收 |
| `net_utils.py` | `get_local_ip()`（UDP 探默认路由）、`get_free_port()` |
| `smoke_test.py` | 分段冒烟测试 |
| `bench/nfs_scan.py` | NFS 带宽扫描（O_DIRECT，与插件解耦） |
| `bench/e2e_vllm_load.py` | 端到端：vLLM 经插件加载 vs 从共享盘加载 |

### 2.3 数据模型：bucket

这是理解全部代码的关键。**一个 tensor 不会单独传输。**

`ModelManager` 把若干 tensor 顺序打包进固定容量的 **bucket**，每个 bucket 是一块连续内存；元数据里每个 tensor 记录的是它在 bucket 内的**字节偏移**。

```
bucket_0  (capacity = bucket_size_bytes, 例如 512 MiB)
┌──────────────────────────────────────────────────────┐
│ embed.weight │ layer.0.weight │ layer.0.bias │ ... │ unused │
└──────────────────────────────────────────────────────┘
 base_ptr ─┘                                          └─ used_bytes
      ↑ 每个 tensor 记 offset / shape / dtype / nbytes
```

所以：**传输粒度是 bucket（大块连续），yield 粒度是 tensor（切片视图）。**

### 2.4 控制面与数据面分离

| | 控制面 | 数据面 |
| --- | --- | --- |
| 协议 | ZMQ REQ/REP | TCP（可换 mooncake） |
| 端口 | `--metadata-port` | `TcpTransferBackend` 绑定的临时端口，由 `peer_sid()` 对外公布 |
| 内容 | `{model_id}` → `{ok, peer_sid, weight_info}` | `(addr, len)` → 原始字节 |
| 频率 | 每次加载 1 次 | 每 bucket 1 次 |

**控制面只回元数据，权重数据一律走数据面。**

### 2.5 为什么默认 TCP 而不是 mooncake

本机 mooncake 是 CANN 构建，`TransferEngine.initialize()` 会**无条件**安装 `AscendDirectTransport`。一旦装上，`batch_register_memory` 注册的内存就全归它管，传输走 ADXL 设备链路。ADXL 的 `Connect` 对同主机同 device 的对端直接返回 `PARAM_INVALID (103900)`，导致 host→host 传输无法建立。

RDMA 同样走不通：注册会被 Ascend transport 抢走，轮不到 RDMA transport。

TCP 后端完全绕开 NPU 驱动，同时也避开了 devmm 的分钟级阻塞。`MooncakeTransferBackend` 保留在代码里，供 Ascend 路径正常的主机使用。

---

## 3. 完整加载流程与函数调用链

### 3.1 服务端启动链

```
python -m static_weight_cache.server \
    --model-id qwen3-8b --model-path /data/.../qwen3-8B \
    --bind-ip 172.26.4.186 --metadata-port 5599 \
    --capacity-gb 18 --bucket-size-mb 512 --transfer-backend tcp
│
└─ server.main()                                          server.py:161
   ├─ build_arg_parser().parse_args()                     server.py:141
   ├─ TransferBackendConfig(backend, protocol, device_name)   transfer_backend.py:92
   ├─ HostMemConfig(capacity_bytes, pin_memory, lock_memory)  host_mem_manager.py:54
   └─ WeightCacheServer(config)                           server.py:52
      ├─ self.ip = config.bind_ip or get_local_ip()       net_utils.py:8
      ├─ HostMemManager(host_mem_config)                  host_mem_manager.py:85
      │   └─ torch.empty(capacity + alignment, pin_memory=True)     ★ 见 3.3.1
      │       └─ self._pool = _pool_owner.narrow(0, aligned_offset, capacity)
      ├─ ModelManager(host_mem_manager, bucket_size_bytes)   model_manager.py:34
      └─ build_transfer_backend(self.ip, config.transfer)    transfer_backend.py:367
          └─ TcpTransferBackend(local_ip, config)

server.start()                                            server.py:69
   ├─ transfer_backend.start()                            transfer_backend.py:223
   │   ├─ socket.bind((local_ip, 0)); listen(64)          ← 临时端口
   │   └─ Thread(_accept_loop).start()
   ├─ _start_control_socket()                             server.py:119
   │   ├─ metadata_port = get_free_port(ip)               net_utils.py:18
   │   ├─ zmq.Context().socket(zmq.REP).bind(tcp://ip:port)
   │   └─ Thread(_serve_control_loop).start()
   └─ load_model(model_id, model_path, revision)          server.py:91
       ├─ ModelManager.load_safetensors_dir(...)           model_manager.py:125
       │   ├─ sorted(Path(model_path).glob("*.safetensors"))
       │   └─ load_from_named_tensors(...)                 model_manager.py:47
       │       ├─ HostMemManager.allocate_bucket(bucket_size_bytes)   host_mem_manager.py:134
       │       └─ ModelManager._copy_tensor_to_bucket(...)            model_manager.py:150
       └─ transfer_backend.register_buckets(allocations)   transfer_backend.py:127
           └─ TcpTransferBackend.register_regions(...)     transfer_backend.py:305
```

### 3.2 客户端拉取链

```
StaticWeightCacheClient(cache_endpoint, model_id, bucket_size, ...)   client.py:53
   ├─ self.local_ip = local_ip or get_local_ip()           client.py:68
   ├─ build_transfer_backend(local_ip, config)             transfer_backend.py:367
   ├─ transfer_backend.start()
   └─ _resize_recv_buf(bucket_size)                        client.py:78      ★ 见 3.3.5
       ├─ torch.empty(capacity, dtype=uint8, pin_memory=True)
       └─ register_regions([(ptr, numel)])

client.receive_weights()   ← async generator                  client.py:92
   ├─ _query_weight_info(model_id)                         client.py:133
   │   └─ ZMQ REQ: send {model_id} → recv {ok, peer_sid, weight_info}
   └─ for bucket_idx in range(bucket_num):
       ├─ _resize_recv_buf(capacity)                       client.py:78
       ├─ transfer_backend.read(peer_sid, dst_ptr, src_ptr, nbytes)   client.py:107
       │   └─ TcpTransferBackend.read(...)                 transfer_backend.py:310
       │       ├─ _peer(peer_sid)  → socket.create_connection（按 peer 复用）
       │       ├─ conn.sendall(_REQUEST.pack(_OP_READ, src_ptr, nbytes))
       │       ├─ _recv_exact(conn, 9) → (status, actual)
       │       └─ _recv_into_exact(conn, target, actual)   ← recv_into 直接写进 dst
       └─ for name, meta in bucket_meta[i]:
           yield name, recv_buf[offset : offset+size].view(dtype).view(shape)

服务端应答链：
   TcpTransferBackend._accept_loop()                       transfer_backend.py:256
     └─ _serve_conn(conn, addr)                            transfer_backend.py:269
         ├─ _is_served(src_ptr, nbytes)                    transfer_backend.py:298   ★ 见 3.3.4
         ├─ (ctypes.c_uint8 * nbytes).from_address(src_ptr)
         ├─ conn.sendall(_REPLY_HEADER.pack(0, nbytes))
         └─ conn.sendall(memoryview(source))
```

### 3.3 关键函数解析

#### 3.3.1 `HostMemManager.__init__` / `allocate_bucket` — `host_mem_manager.py:85 / :134`

```python
self._pool_owner = torch.empty(capacity + alignment, pin_memory=True)
self._pool = self._pool_owner.narrow(0, aligned_offset, capacity)  # 4 KiB 对齐
```

- 一次性申请整块池子，`narrow` 出一个 4 KiB 对齐的窗口；`base_ptr` 就是对外公布的基址。
- `allocate_bucket` 是**纯线性 bump 分配**：只递增 `_next_offset`，不回收、不合并、不淘汰。池子耗尽直接抛 `MemoryError`。
- **pinned 是刻意的默认值**：池子本身就是缓存，pinned 页不可回收，不会在内存压力下被静默换出；同时 pinned 也正是将来 RDMA 注册所需要的形式。
- 代价：在这台机器上 `pin_memory=True` 会走 NPU 驱动的 devmm 路径，首次触摸设备内存可能**阻塞数分钟**（实测 156 s 一次，日志可见）。所以分配是计时并打日志的。

#### 3.3.2 `ModelManager.load_from_named_tensors` — `model_manager.py:47`

决定"safetensors 的哪个 tensor 进哪个 bucket"的唯一地方：

```python
for name, tensor in named_tensors:
    nbytes = tensor.numel() * tensor.element_size()
    if nbytes > self.bucket_size_bytes:          # 超大 tensor 独占一个 bucket
        ... allocate_bucket(nbytes) ...
    elif current_offset + nbytes > capacity:     # 装不下就切
        flush_bucket(); allocate_bucket(bucket_size_bytes)
    ...
```

打包顺序 = `sorted(glob("*.safetensors"))` 的文件名字典序 + 文件内 `handle.keys()` 顺序 + 贪心填充。

**这个映射没有任何语义**：不是按 layer 分组，不是一文件一 bucket。bucket 边界可以横跨两个 safetensors 文件，一个文件也可以被切成多个 bucket。

> 对比 flexfetch：那边用 `deterministic_nbytes` 先按 `rollout_dtype` 预算好大小，保证**同一 layer 在不同版本间落在相同偏移**，以便多版本复用地址。我们按实际 dtype 现算，没有这个稳定性保证。

#### 3.3.3 `ModelMetadata.to_weight_info` — `protocol.py:65`

控制面返回给客户端的结构，是两端的契约：

```python
{"bucket_num":    N,
 "bucket_meta":   [{tensor_name: {shape, dtype, offset, nbytes}}, ...],
 "bases":         [bucket 基址, ...],
 "capacities":    [bucket 容量, ...],
 "used_bytes":    [实际用掉, ...]}
```

`bases` 里是**服务进程的虚拟地址**，客户端原样回传用于寻址。

#### 3.3.4 `TcpTransferBackend._is_served` — `transfer_backend.py:298`

```python
def _is_served(self, src_ptr, nbytes):
    end = src_ptr + nbytes
    return any(src_ptr >= base and end <= base + capacity
               for base, capacity in self._regions)
```

**这是数据面的安全边界。** 请求里带的是裸地址，如果不校验就等于给任何能连上来的进程一个任意内存读。只有完整落在已登记区域内的区间才允许读，否则回错误码并断开连接。

#### 3.3.5 `StaticWeightCacheClient._resize_recv_buf` — `client.py:78`

```python
self.recv_buf = torch.empty(capacity, dtype=torch.uint8,
                            device=self.recv_device, pin_memory=(self.recv_device == "cpu"))
self.transfer_backend.register_regions([(ptr, numel)])
```

两个要点：

1. **必须注册**。引擎只能写进它被告知过的区域。参考实现 flexfetch 有 `te.register_memory(recv_buf)`，本插件最初漏了这一步，导致传输在建立连接阶段就失败。
2. **扩容后要重新注册**。bucket 容量大于当前缓冲区时重新分配，旧注册随之失效。

#### 3.3.6 `TcpTransferBackend.read` — `transfer_backend.py:310`

```python
conn, lock = self._peer(peer_sid)      # 每个 peer 复用一条连接
with lock:
    conn.sendall(_REQUEST.pack(_OP_READ, src_ptr, nbytes))
    status, actual = _REPLY_HEADER.unpack(_recv_exact(conn, 9))
    target = (ctypes.c_uint8 * actual).from_address(dst_ptr)
    _recv_into_exact(conn, target, actual)
```

- **连接复用**：按 `peer_sid` 缓存连接并加锁。每个 bucket 重新握手会主导整个传输时间。
- **`recv_into` 直接写进目标缓冲区**，中间不落地，少一次拷贝。
- 分帧：请求 `>BQQ` = (op, src_ptr, nbytes)；应答 `>BQ` = (status, nbytes) + 原始字节。

#### 3.3.7 `sync_iter` — `bench/e2e_vllm_load.py`

vLLM 的 `load_weights` 要同步可迭代对象，而插件接口是异步生成器。用一个私有事件循环逐步驱动即可 —— 客户端内部本来就是阻塞式读，这只是调用约定上的适配，不改变执行顺序。

### 3.4 vLLM 侧调用链

插件产出的正是 `Iterable[(name, tensor)]`，与 vLLM 默认 loader 产出的形式完全一致，所以两者可以直接替换、其余代码不动。

```
model.load_weights(sync_iter(client.receive_weights()))
└─ Qwen3ForCausalLM.load_weights(weights)              qwen3.py:338
   └─ AutoWeightsLoader.load_weights(weights)          models/utils.py:398
      ├─ _groupby_prefix(weights)                      models/utils.py
      │   └─ itertools.groupby —— 惰性，只向前多取 1 个元素
      └─ _load_module → _load_param                    models/utils.py
         └─ default_weight_loader(param, weight_data)  weight_utils.py:1231
            └─ param.data.copy_(loaded_weight)         ← 同步 H2D，无 non_blocking
```

`default_weight_loader` 是同步拷贝（没有 `non_blocking=True`），这是当前 `recv_buf` 复用方案能成立的前提：权重在被覆盖前已经拷进显存。

---

## 4. 效果测试

### 4.1 配置

**硬件 / 系统**

| 项 | 值 |
| --- | --- |
| 平台 | openEuler aarch64，Linux 5.10，容器内 |
| NPU | Ascend 910，使用 device 0，单卡 |
| CPU / 内存 | 640 vCPU / 2 TiB |
| 网卡 | 使用 `172.26.4.186`（100 GbE 快卡） |

**软件**

| 项 | 版本 |
| --- | --- |
| Python | 3.12.13 |
| torch | 2.10.0+cpu |
| torch_npu | 2.10.0.post2 |
| vllm | 0.23.1rc1.dev1002+g822865845（editable，`/vllm-workspace/vllm`） |
| vllm_ascend | 0.19.1rc2.dev1043（`/vllm-workspace/vllm-ascend`） |
| mooncake | CANN 9.0.0 自带（本场景未使用） |
| verl | **未安装** |

**模型**

| 项 | 值 |
| --- | --- |
| 路径 | `/data/y50063564/qwen3-8B-dfly/qwen3-8B` |
| 架构 | `Qwen3ForCausalLM`，36 层，hidden 4096，bf16 |
| 权重 | 5 个 safetensors，399 个 tensor，**15.26 GiB** |
| 参数量（已加载） | 291 个 parameter |

**vLLM**

```python
EngineArgs(model=..., tensor_parallel_size=1, enforce_eager=True,
           max_model_len=2048, gpu_memory_utilization=0.5)
init_worker_distributed_environment(vllm_config, rank=0, local_rank=0, backend="hccl")
```

**插件**

```bash
python -m static_weight_cache.server \
    --model-id qwen3-8b \
    --model-path /data/y50063564/qwen3-8B-dfly/qwen3-8B \
    --bind-ip 172.26.4.186 --metadata-port 5599 \
    --capacity-gb 18 --bucket-size-mb 512 --transfer-backend tcp
```

| 项 | 值 |
| --- | --- |
| backend | `tcp` |
| 内存池 | 18 GiB，**pinned**（分配耗时 1.45 s） |
| bucket | 512 MiB → 打包成 **30 个 bucket** |
| 控制面 | ZMQ，`172.26.4.186:5599` |
| 数据面 | TCP 临时端口 |
| client recv_buf | 512 MiB pinned |

### 4.2 结果

```
==================================================================
  disk   (NFS -> vLLM)   :     5.70 s
  plugin (cache -> vLLM) :     6.32 s
  speedup                :     0.90 x
  weights identical      : YES
==================================================================
```

**正确性**：291 个参数**逐个 sha256 对比，全部逐字节一致**。

自检：清零后 291 个参数的摘要**全部**改变，证明"两边一致"不是因为校验检测不出变化。

**服务端一次性冷加载**：14 s（15.26 GiB，等效 1.17 GB/s，与 NFS 单流实测吻合）。

### 4.3 分析：为什么当前是负收益

**原因一：磁盘基线被 page cache 喂饱了。**

缓存服务进程刚把整个模型从 NFS 读进池子，同时也填满了 OS page cache。所以 vLLM 的"从盘加载"实际是内存速度：

```
vLLM 内部计时: Loading weights took 4.38 seconds  (15.26 GiB)
→ 3.5 GiB/s，远超 NFS 实测上限 1.6 GiB/s
```

**原因二：插件路径多一跳拷贝。**

```
disk   : page cache → mmap → copy_ 到 NPU
plugin : server host 池 → TCP loopback → client pinned buf → copy_ 到 NPU
```

因此真实的三方对比是：

| 场景 | 耗时 |
| --- | --- |
| vLLM 直接从 NFS 加载（**page cache 未命中**） | ≈ 14 s NFS 读 + H2D ≈ **15–16 s** |
| vLLM 从插件加载（热） | **6.32 s** |
| vLLM 直接从 NFS 加载（**page cache 已命中**） | **5.70 s** |

**结论：插件在"单机 + page cache 已命中"场景下没有收益，反而多一跳。** 这个结论也说明"只测 server→client 搬运"是不够的 —— 那样完全看不出问题。

### 4.4 插件的价值场景（本次均未测到）

1. **page cache 未命中**（真冷启动）—— 15–16 s vs 6.32 s
2. **多进程 / 多节点并发** —— 避免 N 份重复的 NFS 读（NFS 上限 1.6 GiB/s 是共享的）
3. **模型规模超过 page cache 可容纳的量**（本机 2 TiB 内存，15 GiB 模型必然被缓存）

---

## 5. 已知缺口

### 5.1 功能缺口

| 缺口 | 说明 |
| --- | --- |
| 无 RR 分桶 | client 无条件拉取全部 bucket，没有 rank/world_size 过滤。N 个 worker 会各拉一份全量 |
| 无组内广播 | flexfetch 是 rank 0 拉取后经 HCCL 组内广播；本插件每个 worker 都是独立 client，TP=8 会产生 8 份完整拷贝 |
| 无 H2D | client yield 的是 CPU tensor 视图，显存搬运由调用方（vLLM loader）自己做 |
| 无版本 / 不能重载 | `ModelManager.load_from_named_tensors` 遇到同 `model_id` 直接抛 `ValueError`；`HostMemManager` 从不回收 |
| 无认证 | 控制面与数据面均无认证，任何能连上的进程都能读取缓存内容 |

### 5.2 实现隐患

- **`recv_buf` 别名**：`client.receive_weights()` 全程复用同一块 `recv_buf`，yield 出去的是它的**视图**。下一个 bucket 读进来会就地覆盖上一个 bucket 已 yield 的内容。当前之所以安全，是因为 `default_weight_loader` 是同步拷贝（3.4 节）；一旦调用方改成异步拷贝或缓存 tensor 引用，就会静默读到脏数据。
- **末尾 bucket 多传**：client 读的是 `capacity` 而不是 `used_bytes`（`client.py:111`），每个 bucket 都按满容量传，最后一组通常只用了一部分，白传。
- **`--lock-memory` 与 pinned 冗余**：pinned 内存已被分配器锁定，`lock_memory` 只在 pageable 时才有意义（代码已处理并说明）。
- **NPU 驱动阻塞**：本机上任何 devmm 交互（分配、析构）都可能阻塞数分钟且 `kill -9` 无效。服务端因为要申请 pinned 池，必然要付这个代价。

---

## 6. 后续建议

按投入产出排序：

1. **真冷基线**：加载前对模型文件做 `posix_fadvise(DONTNEED)`（只针对这几个文件，不动全局 page cache），重测 disk 基线，才能得到有意义的加速比。
2. **多并发**：N 个 vLLM worker 同时加载，观察 NFS 共享上限下插件侧的收益。
3. **接 `init_process_group` + 组内广播**：支持 TP/DP 的真实形态。注意本机未装 verl，需要先解决环境。
4. **消除一次拷贝**：`TCP → CPU pinned buf → copy_ 到 NPU` 中的中间一跳；RDMA 可直接写显存，这也是保持当前数据流不变的价值所在。
5. **放开 NFS slot table**：`sysctl -w sunrpc.tcp_slot_table_entries=64`（运行时生效、可逆），验证能否接近 70–80 Gbps。
6. **补 `HostMemManager` 的版本与回收机制**，支撑"重复加载同一模型"。

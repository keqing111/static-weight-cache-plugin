#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""静态权重缓存的单进程 host 内存池。

这个模块只做一件事：一次性申请一大块长驻 host 内存，然后按 bucket 线性切分出去。
服务进程持有它，所以分配出来的内存不会随 vLLM 进程重启而消失 —— 这正是"避免
反复从共享盘拉权重"能成立的前提。

Ascend 主机上的内存约束
-----------------------
Mooncake 的 Ascend transport 由 ``TransferEngine.initialize()`` 无条件安装，而它
只接受自己认得出来的缓冲区。普通匿名内存（哪怕已经 ``mlock`` 过）会被判定为
``location:*``，``batch_register_memory`` 直接返回 -1。所以池子必须来自 Ascend 的
锁页内存分配器，也就是 ``pin_memory=True``。

该分配器实际落在 NPU 驱动的 devmm 模块里。进程内**第一次**触摸设备内存会触发一次
``devmm_setup_device_proc`` ioctl，它阻塞在不可中断睡眠（D 状态）里：本机实测过一次
约 156 秒，期间 ``kill -9`` 无效，看起来就像卡死。同一进程后续的分配是瞬时的。
所以这里的分配是计时并打日志的，而不是默默卡住。

也因此，``mlock`` **不能**替代 pin：它只是阻止页被换出，并不会让缓冲区变得可注册。
它只对 pageable 的情形有意义，而那种情形下它也确实是唯一能做的事。
"""

from __future__ import annotations

import ctypes
import logging
import time
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


def align_up(value: int, alignment: int) -> int:
    """把 value 向上取整到 alignment 的整数倍。

    公式 ``(x + a - 1) // a * a`` 是向上取整的标准写法：先加上 a-1 保证"只要有
    零头就进位"，整除丢掉零头，再乘回去还原量级。
    """
    return (value + alignment - 1) // alignment * alignment


@dataclass
class HostMemConfig:
    """内存池的构造参数。"""

    capacity_bytes: int
    # **地址对齐粒度**，默认 4096 字节（4 KiB）。注意它和 bucket 大小是两个完全
    # 不同的旋钮，不要混淆：alignment 只决定"每块内存的首地址落在什么边界上"，
    # 不决定每块切多大；后者是 ModelManager 的 bucket_size_bytes。
    #
    # 取 4 KiB 的理由：等于页大小（mlock 要求页对齐），同时整除 NFS 的 1 MiB rsize，
    # 也是 TransferEngine 对可注册内存首地址要求的公约数（HCCS 要 2 MB，RDMA 要 4 KB）。
    alignment: int = 4096
    # 凡是交给传输引擎注册的池子都要保持 True。置 False 得到的池子在 Ascend 主机上
    # 根本无法注册，只能用于完全不碰传输引擎的测试。
    pin_memory: bool = True
    # 仅在 pin_memory=False 时才有意义：pinned 内存已被分配器本身锁住，再 mlock 一次
    # 是多余的。它不是 pin 的替代品。
    lock_memory: bool = False


@dataclass
class BucketAllocation:
    """从池子里切出来的一块 bucket。

    这是"内存池"和"模型打包"两层之间的唯一接口：ModelManager 只管往里写，
    传输层只管拿 ``base_ptr``/``capacity`` 去注册和寻址。
    """

    bucket_id: int
    offset: int          # 相对池子基址的字节偏移
    capacity: int        # 这块 bucket 的容量（已按 alignment 向上取整）
    used_bytes: int      # 实际写入了多少字节
    buffer: torch.Tensor  # 指向池子内该区间的视图（不是拷贝）

    @property
    def base_ptr(self) -> int:
        """这块 bucket 的首地址。

        注意这是**服务进程的虚拟地址**，它会通过元数据原样发给客户端，客户端后续
        再用它来寻址。所以它既是本地指针，也是跨进程的寻址标识。
        """
        return int(self.buffer.data_ptr())


class HostMemManager:
    """在一块长驻 host 缓冲区上实现的极简线性分配器。

    V1 刻意不做 free list、不做碎片整理、不做淘汰、不支持模型复用 —— 分配指针只
    向前走。服务进程持有本对象，所以分配出来的内容能跨 vLLM 重启存活。

    代价是：池子耗尽只能报错，不能回收；同一个 model_id 也无法二次加载。
    """

    def __init__(self, config: HostMemConfig) -> None:
        if config.capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be positive")
        self.config = config

        # 不 pin 的话池子就无法交给传输引擎。这里只警告不报错，因为确实存在
        # "只想验证内存池本身、不碰传输"的测试场景。
        if not config.pin_memory:
            logger.warning(
                "HostMemManager built without pin_memory: the transfer engine cannot register "
                "this pool (Ascend transport reports the buffers as location:* and refuses them). "
                "Use this only for tests that never touch the transfer engine."
            )

        # pinned 分配在繁忙的 Ascend 主机上可能阻塞数分钟（见模块开头说明），
        # 所以计时并显式打日志，否则从外部看就是"卡住了"。
        # pageable 只是一次普通的匿名映射，永远不碰驱动，因此不做这类提醒。
        if config.pin_memory:
            logger.info(
                "Allocating %.2f GB pinned host pool; the first device-memory touch in this "
                "process may block for minutes on Ascend hosts",
                config.capacity_bytes / (1 << 30),
            )
        else:
            logger.info("Allocating %.2f GB pageable host pool", config.capacity_bytes / (1 << 30))
        start = time.time()

        # 为什么多申请 alignment 个字节？
        #
        # 因为 torch.empty() 没有对齐参数，也没有对齐承诺：它的 CPU 分配器只保证
        # 64 字节对齐（为 SIMD 向量化用），pinned 分配在实践中通常落在页边界上，
        # 但那是"通常"，API 层面并不保证。所以我们拿不到"直接申请一块对齐内存"
        # 这个选项，只能多要一点、自己裁掉头部。
        #
        # 为什么必须对齐？传输引擎注册可访问内存时要求基址对齐（flexfetch 里的
        # 注释写得很明确：HCCS 协议要 2 MB，RDMA 协议要 4 KB）。基址不满足要求时
        # batch_register_memory 可能直接失败。另外 mlock 也要求起始地址页对齐。
        # 注意：这对当前的 TCP 后端没有影响（TCP 不注册内存、不对齐也没关系），
        # 纯属为 mooncake/RDMA 路径留的正确性。
        #
        # 代价是最多浪费 alignment-1 字节，对几 GB 的池子可以忽略。
        self._pool_owner = torch.empty(
            config.capacity_bytes + config.alignment,
            dtype=torch.uint8,
            device="cpu",
            pin_memory=config.pin_memory,
        )
        logger.info("Host pool allocated in %.2fs", time.time() - start)

        # 把"起始地址任意、长 capacity+alignment 的内存"转换成"起始地址严格对齐、
        # 长 capacity 的内存"：
        #   align_up(raw_ptr, a)   >= raw_ptr 的最小 a 的倍数，即第一个对齐地址
        #   aligned_offset         = 上面那个地址距 raw_ptr 的字节数，落在 [0, a-1]
        #   narrow(0, off, cap)    torch 的切片，返回**视图**不拷贝，从 off 处取 cap 字节
        #
        # _pool 是视图，所以 _pool_owner 必须一直持有；否则底层内存被释放，视图
        # 就成了野指针。
        raw_ptr = int(self._pool_owner.data_ptr())
        aligned_offset = align_up(raw_ptr, config.alignment) - raw_ptr
        self._pool = self._pool_owner.narrow(0, aligned_offset, config.capacity_bytes)

        self._next_offset = 0        # 线性分配指针：下一个空闲字节的位置
        self._next_bucket_id = 0     # 递增的 bucket 编号
        # bucket_id -> 该 bucket 的分配记录。
        #
        # 键是 bucket_id（从 0 开始的递增编号），**不是地址、也不是偏移**。它存在的
        # 意义是把"序列化出去的元数据"和"进程内的实际分配"对起来：
        #   BucketAllocation.bucket_id → 写进 BucketMetadata → 随元数据发给客户端
        #   服务端 load_model 里再用 get_bucket(bucket.bucket_id) 反查回真实分配
        # 注意 bucket_id 只在本进程内有效；客户端不会用它寻址（客户端用 base_ptr）。
        self._allocations: dict[int, BucketAllocation] = {}
        self._locked = False

        if config.lock_memory:
            if config.pin_memory:
                # pinned 内存本来就是锁住的，重复 mlock 没有意义。
                logger.debug("lock_memory ignored: pinned memory is already locked by the allocator")
            else:
                self._lock_memory()

    @property
    def base_ptr(self) -> int:
        """对齐后的池子基址（所有 bucket 偏移都相对它计算）。"""
        return int(self._pool.data_ptr())

    def allocate_bucket(self, capacity: int, used_bytes: int | None = None) -> BucketAllocation:
        """切出一块 capacity 字节的 bucket。纯 bump 分配。

        所谓 **bump 分配**（bump pointer allocation），就是分配器只维护一个"当前
        位置"指针，分配 = "取当前指针的值，然后把指针往前推 size 字节"。没有空闲
        链表、不搜索空洞、不合并、不回收 —— free 要么不存在，要么只能整块重置。
        它是理论上最快的分配器（成本就是一次加法），代价是永不回收。

        切的位置是**已用区的尾部**（used/free 的分界线），不是整个池子的尾部：
        已用区从池子头部往尾部生长，新 bucket 紧贴在上一块后面，池子最末尾的那些
        字节反而是最后才会被用到的。体现这一点的就是下面这两行::

            offset = self._next_offset        # 取当前前沿
            self._next_offset += capacity     # 前沿再往后推 capacity

        池子不够就直接抛 MemoryError —— V1 没有淘汰机制，这是设计如此。
        """
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        capacity = align_up(capacity, self.config.alignment)
        # 边界检查：推进之后不能越过池子总容量。
        if self._next_offset + capacity > self.config.capacity_bytes:
            raise MemoryError(
                f"Static weight cache exhausted: requested={capacity}, "
                f"free={self.config.capacity_bytes - self._next_offset}"
            )

        bucket_id = self._next_bucket_id
        self._next_bucket_id += 1
        offset = self._next_offset
        self._next_offset += capacity
        # 真正"切出来"的一步：narrow 返回视图，不拷贝数据。
        view = self._pool.narrow(0, offset, capacity)
        allocation = BucketAllocation(
            bucket_id=bucket_id,
            offset=offset,
            capacity=capacity,
            # 调用方可以在分配时先声明"这块会用多少"，稍后写数据时再被更新。
            used_bytes=used_bytes if used_bytes is not None else capacity,
            buffer=view,
        )
        self._allocations[bucket_id] = allocation
        return allocation

    def get_bucket(self, bucket_id: int) -> BucketAllocation:
        """按编号取回已分配的 bucket。"""
        return self._allocations[bucket_id]

    def write_bytes(self, allocation: BucketAllocation, offset: int, data: bytes) -> None:
        """往 bucket 的指定偏移写入一段字节。

        注意：ModelManager 走的是自己那条更快的 tensor 拷贝路径，并不调用这里。
        本方法主要留给测试和调试用。
        """
        end = offset + len(data)
        if end > allocation.capacity:
            raise ValueError(f"Write exceeds bucket capacity: end={end}, capacity={allocation.capacity}")
        src = torch.frombuffer(bytearray(data), dtype=torch.uint8)
        allocation.buffer[offset:end].copy_(src)
        # used_bytes 只增不减，代表"这块 bucket 里有效数据的右边界"
        allocation.used_bytes = max(allocation.used_bytes, end)

    def get_registered_regions(self) -> list[BucketAllocation]:
        """返回当前所有已分配的 bucket。"""
        return list(self._allocations.values())

    def stats(self) -> dict[str, int | bool]:
        """给日志和监控用的池子状态快照。"""
        return {
            "capacity_bytes": self.config.capacity_bytes,
            "used_bytes": self._next_offset,
            "free_bytes": self.config.capacity_bytes - self._next_offset,
            "bucket_count": len(self._allocations),
            "pin_memory": self.config.pin_memory,
            "locked": self._locked,
        }

    def close(self) -> None:
        """释放资源。只负责解锁，内存本身随对象一起被回收。"""
        if self._locked:
            self._unlock_memory()

    def _lock_memory(self) -> None:
        """把池子锁进物理内存，防止被换出。

        注意这**不会**让池子变得可注册 —— 见模块开头关于 ``location:*`` 的说明。
        """
        # mlock 要求起始地址页对齐。池子基址对齐到 config.alignment，所以调用方
        # 必须保证 alignment 不小于页大小（默认 4096 满足）。
        # 长度向上取整，因为 mlock 只锁整页。
        length = align_up(self.config.capacity_bytes, self.config.alignment)
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        ret = libc.mlock(ctypes.c_void_p(self.base_ptr), ctypes.c_size_t(length))
        if ret != 0:
            errno = ctypes.get_errno()
            # 带上具体数值和可行的解法，否则只看到一个裸 errno 很难排查。
            raise OSError(
                errno,
                f"mlock failed for static weight cache pool: {errno} "
                f"(length={length}, capacity={self.config.capacity_bytes}). "
                f"Raise RLIMIT_MEMLOCK with `ulimit -l unlimited`, or run with CAP_IPC_LOCK. "
                f"Note that locking does not make the pool registrable with the transfer engine.",
            )
        self._locked = True
        logger.info("Locked static weight cache pool: %s bytes", length)

    def _unlock_memory(self) -> None:
        """解除锁定。失败只记警告 —— 进程退出时内核自然会回收。"""
        length = align_up(self.config.capacity_bytes, self.config.alignment)
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        ret = libc.munlock(ctypes.c_void_p(self.base_ptr), ctypes.c_size_t(length))
        if ret != 0:
            errno = ctypes.get_errno()
            logger.warning("munlock failed for static weight cache pool: errno=%s", errno)
        self._locked = False

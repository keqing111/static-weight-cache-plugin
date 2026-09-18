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
"""静态权重缓存的服务进程。

这是整个插件里唯一常驻的角色。它做四件事：

1. 申请一块 host 内存池（见 ``host_mem_manager``）；
2. 把模型从共享盘读进这块池子（只读一次，见 ``model_manager``）；
3. 通过控制面回答"某个模型有哪些 bucket、在什么地址"；
4. 通过数据面把 bucket 的字节发给客户端。

进程模型
--------
本进程有两个后台线程，互不干扰：

* ``_serve_control_loop``：ZMQ REP，回元数据。轻量、低频（每次加载只问一次）。
* 数据面线程由具体的 ``TransferBackend`` 提供（TCP 后端是 accept + 每连接一个
  handler）。

**控制面只回元数据，权重数据一律走数据面**，两条通道的端口也是分开的。这样权重
传输的大流量不会阻塞元数据查询，反之亦然。

为什么服务进程要常驻
--------------------
池子里的权重是从共享盘读一遍来的。vLLM 进程反复重启时，只要本进程还活着，就
不需要再碰共享盘 —— 这正是插件要解决的问题。所以 ``HostMemManager`` 由本对象
持有，生命周期与进程一致，而不是与某次加载一致。
"""

from __future__ import annotations

import argparse
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

import zmq

from .host_mem_manager import HostMemConfig, HostMemManager
from .model_manager import ModelManager
from .net_utils import get_free_port, get_local_ip
from .protocol import CacheQuery, CacheResponse
from .transfer_backend import TransferBackend, TransferBackendConfig, build_transfer_backend

logger = logging.getLogger(__name__)


@dataclass
class WeightCacheServerConfig:
    """服务进程的全部配置。"""

    model_id: str
    # 启动时就加载的模型路径。留空则只起服务、不预加载，之后用 load_model/
    # load_named_tensors 手动加载。
    model_path: str | None = None
    revision: str | None = None
    # 绑定哪个网卡。留空走 get_local_ip()（注意它跟的是默认路由，多网卡机器上
    # 未必是你想要的那张，本机就会解析到慢卡）。
    bind_ip: str | None = None
    # 控制面端口。0 表示自动挑一个空闲端口。
    metadata_port: int = 0
    capacity_bytes: int = 8 << 30
    # 传给 ModelManager 的 bucket 容量上限。
    bucket_size_bytes: int = 1 << 30
    # 不传则按 capacity_bytes 现造一个 HostMemConfig。
    host_mem: HostMemConfig | None = None
    transfer: TransferBackendConfig = field(default_factory=TransferBackendConfig)


class WeightCacheServer:
    """持有权重内存池，并对外提供元数据查询与数据传输两个服务。"""

    def __init__(self, config: WeightCacheServerConfig) -> None:
        self.config = config
        self.ip = config.bind_ip or get_local_ip()

        # 池子先建。注意这一步可能阻塞数分钟（pinned 分配要走 NPU 驱动的 devmm），
        # 所以构造本对象就可能有明显的等待。
        host_mem_config = config.host_mem or HostMemConfig(capacity_bytes=config.capacity_bytes)
        self.host_mem_manager = HostMemManager(host_mem_config)

        # ModelManager 只负责往池子里写并产出元数据，不持有池子本身。
        self.model_manager = ModelManager(self.host_mem_manager, config.bucket_size_bytes)

        # 数据面。构造时只记录配置，真正的监听在 start() 里才开始。
        self.transfer_backend: TransferBackend = build_transfer_backend(self.ip, config.transfer)

        self._context: zmq.Context | None = None
        self.control_socket = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.metadata_port = config.metadata_port

    @property
    def metadata_endpoint(self) -> str:
        """控制面的 ZMQ 地址，客户端用它来查元数据。"""
        return f"tcp://{self.ip}:{self.metadata_port}"

    def start(self) -> None:
        """起数据面 + 控制面，然后按配置预加载模型。

        顺序很重要：数据面必须先于加载，因为 register_buckets 需要后端已经就绪；
        而控制面必须先于加载对外可用，否则客户端可能查到"模型不存在"。
        """
        self.transfer_backend.start()
        self._start_control_socket()
        if self.config.model_path:
            self.load_model(self.config.model_id, self.config.model_path, self.config.revision)

    def serve_forever(self) -> None:
        """阻塞在控制面线程上，直到 stop() 被调用。"""
        if self._thread is None:
            raise RuntimeError("Server has not been started")
        self._thread.join()

    def stop(self) -> None:
        """关闭控制面与数据面，并释放内存池。"""
        self._stop.set()
        if self.control_socket is not None:
            # linger=0：不要为了发完残留消息而阻塞退出。
            self.control_socket.close(linger=0)
            self.control_socket = None
        if self._context is not None:
            self._context.term()
            self._context = None
        self.transfer_backend.close()
        self.host_mem_manager.close()

    def load_model(self, model_id: str, model_path: str | Path, revision: str | None = None) -> None:
        """从目录里的 safetensors 加载模型，并把占用的 bucket 交给数据面登记。

        "登记"这一步不能省：TCP 后端只允许读取已登记范围内的地址（裸地址协议
        必须有这个边界），mooncake 后端则必须注册后才能被远端访问。
        """
        metadata = self.model_manager.load_safetensors_dir(model_id=model_id, model_path=model_path, revision=revision)
        allocations = [self.host_mem_manager.get_bucket(bucket.bucket_id) for bucket in metadata.buckets]
        self.transfer_backend.register_buckets(allocations)

    def load_named_tensors(self, model_id: str, named_tensors, revision: str | None = None) -> None:
        """同上，但权重来自内存里的 ``(name, tensor)`` 可迭代对象（测试用）。"""
        metadata = self.model_manager.load_from_named_tensors(
            model_id=model_id,
            named_tensors=named_tensors,
            revision=revision,
        )
        allocations = [self.host_mem_manager.get_bucket(bucket.bucket_id) for bucket in metadata.buckets]
        self.transfer_backend.register_buckets(allocations)

    def handle_query(self, payload) -> dict:
        """处理一次元数据查询，返回可直接回给客户端的字典。

        成功时返回 ``{ok, peer_sid, weight_info}``：``peer_sid`` 是**数据面**的地址
        （TCP 后端即 ip:port），客户端拿到后去那里读实际字节；``weight_info`` 描述
        各 bucket 的地址、容量与内含 tensor 的相对偏移。

        任何异常都被吞掉并转成 ``ok=False`` 的响应 —— 控制面线程不能因为一次坏
        请求就退出。
        """
        try:
            query = CacheQuery.from_wire(payload)
            metadata = self.model_manager.get_model(query.model_id)
            response = CacheResponse(
                ok=True,
                peer_sid=self.transfer_backend.peer_sid(),
                weight_info=metadata.to_weight_info(),
            )
        except Exception as err:
            logger.exception("Failed to handle weight cache query")
            response = CacheResponse(ok=False, error=str(err))
        return response.to_wire()

    def _start_control_socket(self) -> None:
        """绑定控制面端口，并在后台线程里开始应答。"""
        if self.metadata_port == 0:
            self.metadata_port = get_free_port(self.ip)
        self._context = zmq.Context()
        # REP 意味着严格的"收到一条、回一条"。元数据查询频次很低（每次模型加载
        # 一次），所以单线程串行处理完全够用，也省掉了并发状态管理。
        self.control_socket = self._context.socket(zmq.REP)
        self.control_socket.bind(self.metadata_endpoint)
        self._thread = threading.Thread(target=self._serve_control_loop, daemon=True)
        self._thread.start()
        logger.info("Static weight cache metadata server listening on %s", self.metadata_endpoint)

    def _serve_control_loop(self) -> None:
        """控制面主循环。daemon 线程，随进程退出而结束。"""
        assert self.control_socket is not None
        while not self._stop.is_set():
            try:
                payload = self.control_socket.recv_pyobj()
                self.control_socket.send_pyobj(self.handle_query(payload))
            except zmq.ZMQError:
                # stop() 里 close() 掉 socket 也会走到这里，那种情况不该记成错误。
                if not self._stop.is_set():
                    logger.exception("ZMQ control socket failed")
                break


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start a static DDR weight cache server")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--bind-ip", default=None)
    parser.add_argument("--metadata-port", type=int, default=0)
    parser.add_argument("--capacity-gb", type=float, default=8.0)
    parser.add_argument("--bucket-size-mb", type=int, default=1024)
    parser.add_argument("--transfer-backend", default="tcp", choices=["tcp", "mooncake"])
    parser.add_argument("--transfer-protocol", default="tcp")
    parser.add_argument("--device-name", default="")
    parser.add_argument("--lock-memory", action="store_true")
    parser.add_argument(
        "--no-pin-memory",
        action="store_true",
        help="pageable pool; the cache then becomes reclaimable and the pool is no longer RDMA-registrable",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = build_arg_parser().parse_args()

    transfer = TransferBackendConfig(
        backend=args.transfer_backend,
        protocol=args.transfer_protocol,
        device_name=args.device_name,
    )
    # 默认 pin。池子本身就是缓存，必须是不可回收的：pageable 的页在内存压力下会被
    # 悄悄换出，缓存就在无人察觉的情况下退化成"每次都要重新从共享盘拉"。而且
    # pinned 也正是将来 RDMA 传输需要注册的形式。
    # 注意在本机 pinned 分配会走 NPU 驱动的 devmm 路径，可能阻塞数分钟。
    pin_memory = not args.no_pin_memory

    capacity_bytes = int(args.capacity_gb * (1 << 30))
    config = WeightCacheServerConfig(
        model_id=args.model_id,
        model_path=args.model_path,
        bind_ip=args.bind_ip,
        metadata_port=args.metadata_port,
        capacity_bytes=capacity_bytes,
        bucket_size_bytes=args.bucket_size_mb << 20,
        host_mem=HostMemConfig(
            capacity_bytes=capacity_bytes,
            pin_memory=pin_memory,
            lock_memory=args.lock_memory,
        ),
        transfer=transfer,
    )
    server = WeightCacheServer(config)
    try:
        server.start()
        logger.info("Static weight cache server ready: %s", server.metadata_endpoint)
        server.serve_forever()
    finally:
        server.stop()


if __name__ == "__main__":
    main()

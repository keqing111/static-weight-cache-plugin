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
"""Static weight cache server process."""

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
    model_id: str
    model_path: str | None = None
    revision: str | None = None
    bind_ip: str | None = None
    metadata_port: int = 0
    capacity_bytes: int = 8 << 30
    bucket_size_bytes: int = 1 << 30
    host_mem: HostMemConfig | None = None
    transfer: TransferBackendConfig = field(default_factory=TransferBackendConfig)


class WeightCacheServer:
    """Owns the static weight memory pool and metadata query socket."""

    def __init__(self, config: WeightCacheServerConfig) -> None:
        self.config = config
        self.ip = config.bind_ip or get_local_ip()
        host_mem_config = config.host_mem or HostMemConfig(capacity_bytes=config.capacity_bytes)
        self.host_mem_manager = HostMemManager(host_mem_config)
        self.model_manager = ModelManager(self.host_mem_manager, config.bucket_size_bytes)
        self.transfer_backend: TransferBackend = build_transfer_backend(self.ip, config.transfer)
        self._context: zmq.Context | None = None
        self.control_socket = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.metadata_port = config.metadata_port

    @property
    def metadata_endpoint(self) -> str:
        return f"tcp://{self.ip}:{self.metadata_port}"

    def start(self) -> None:
        self.transfer_backend.start()
        self._start_control_socket()
        if self.config.model_path:
            self.load_model(self.config.model_id, self.config.model_path, self.config.revision)

    def serve_forever(self) -> None:
        if self._thread is None:
            raise RuntimeError("Server has not been started")
        self._thread.join()

    def stop(self) -> None:
        self._stop.set()
        if self.control_socket is not None:
            self.control_socket.close(linger=0)
            self.control_socket = None
        if self._context is not None:
            self._context.term()
            self._context = None
        self.transfer_backend.close()
        self.host_mem_manager.close()

    def load_model(self, model_id: str, model_path: str | Path, revision: str | None = None) -> None:
        metadata = self.model_manager.load_safetensors_dir(model_id=model_id, model_path=model_path, revision=revision)
        allocations = [self.host_mem_manager.get_bucket(bucket.bucket_id) for bucket in metadata.buckets]
        self.transfer_backend.register_buckets(allocations)

    def load_named_tensors(self, model_id: str, named_tensors, revision: str | None = None) -> None:
        metadata = self.model_manager.load_from_named_tensors(
            model_id=model_id,
            named_tensors=named_tensors,
            revision=revision,
        )
        allocations = [self.host_mem_manager.get_bucket(bucket.bucket_id) for bucket in metadata.buckets]
        self.transfer_backend.register_buckets(allocations)

    def handle_query(self, payload) -> dict:
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
        if self.metadata_port == 0:
            self.metadata_port = get_free_port(self.ip)
        self._context = zmq.Context()
        self.control_socket = self._context.socket(zmq.REP)
        self.control_socket.bind(self.metadata_endpoint)
        self._thread = threading.Thread(target=self._serve_control_loop, daemon=True)
        self._thread.start()
        logger.info("Static weight cache metadata server listening on %s", self.metadata_endpoint)

    def _serve_control_loop(self) -> None:
        assert self.control_socket is not None
        while not self._stop.is_set():
            try:
                payload = self.control_socket.recv_pyobj()
                self.control_socket.send_pyobj(self.handle_query(payload))
            except zmq.ZMQError:
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
    # Pinned by default. The pool *is* the cache, so it must not be reclaimable;
    # a pageable pool can be evicted under memory pressure and quietly turn into
    # a miss. It is also the form an RDMA transport would need to register.
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




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
"""Static weight cache client with a CheckpointEngine-like receive_weights API."""

from __future__ import annotations

import argparse
import logging
import math
import time
from typing import AsyncGenerator

import torch
import zmq

from .net_utils import get_local_ip
from .transfer_backend import TransferBackendConfig, build_transfer_backend

logger = logging.getLogger(__name__)


_DTYPE_BY_NAME = {
    "torch.float32": torch.float32,
    "torch.float": torch.float32,
    "torch.float16": torch.float16,
    "torch.half": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.int64": torch.int64,
    "torch.long": torch.int64,
    "torch.int32": torch.int32,
    "torch.int16": torch.int16,
    "torch.int8": torch.int8,
    "torch.uint8": torch.uint8,
    "torch.bool": torch.bool,
}


class StaticWeightCacheClient:
    """Receives cached model weights as an async generator of named tensors."""

    def __init__(
        self,
        cache_endpoint: str,
        model_id: str,
        bucket_size: int,
        transfer_config: TransferBackendConfig | None = None,
        recv_device: str = "cpu",
        local_ip: str | None = None,
    ) -> None:
        self.cache_endpoint = cache_endpoint
        self.model_id = model_id
        self.bucket_size = bucket_size
        self.recv_device = recv_device
        # get_local_ip() follows the default route, which need not be the NIC we
        # want the data plane on. Local-ip lookup is only a fallback.
        self.local_ip = local_ip or get_local_ip()
        self.transfer_backend = build_transfer_backend(
            self.local_ip,
            transfer_config or TransferBackendConfig(),
        )
        self.transfer_backend.start()
        self.recv_buf: torch.Tensor | None = None
        self._recv_buf_is_registered = False
        self._resize_recv_buf(bucket_size)

    def _resize_recv_buf(self, capacity: int) -> None:
        """Grow the landing buffer and register it with the transfer engine.

        The engine can only write into memory it has registered, and on Ascend
        hosts only pinned memory is accepted. Skipping this leaves the engine
        with an unregistered destination and the read fails at connection time.
        """
        if self.recv_buf is not None and self.recv_buf.numel() >= capacity:
            return
        pinned = self.recv_device == "cpu"
        self.recv_buf = torch.empty(capacity, dtype=torch.uint8, device=self.recv_device, pin_memory=pinned)
        self.transfer_backend.register_regions([(int(self.recv_buf.data_ptr()), int(self.recv_buf.numel()))])
        self._recv_buf_is_registered = True

    async def receive_weights(self, model_id: str | None = None) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        target_model_id = model_id or self.model_id
        response = self._query_weight_info(target_model_id)
        if not response.get("ok", False):
            raise RuntimeError(response.get("error") or f"Cache query failed for model {target_model_id}")

        peer_sid = response["peer_sid"]
        weight_info = response["weight_info"]
        total_bytes = 0
        start_time = time.time()

        for bucket_idx in range(weight_info["bucket_num"]):
            capacity = int(weight_info["capacities"][bucket_idx])
            used_bytes = int(weight_info.get("used_bytes", weight_info["capacities"])[bucket_idx])
            self._resize_recv_buf(capacity)
            self.transfer_backend.read(
                peer_sid=peer_sid,
                dst_ptr=int(self.recv_buf.data_ptr()),
                src_ptr=int(weight_info["bases"][bucket_idx]),
                nbytes=capacity,
            )
            bucket_meta = weight_info["bucket_meta"][bucket_idx]
            total_bytes += used_bytes
            for name, meta in bucket_meta.items():
                dtype = self._parse_dtype(meta["dtype"])
                shape = tuple(meta["shape"])
                size = dtype.itemsize * math.prod(shape)
                offset = int(meta["offset"])
                tensor = self.recv_buf[offset : offset + size].view(dtype=dtype).view(shape)
                yield name, tensor

        elapsed = max(time.time() - start_time, 1e-9)
        logger.info(
            "Received static cached weights for %s: %.2f GB/s",
            target_model_id,
            total_bytes / elapsed / (1024 * 1024 * 1024),
        )

    def close(self) -> None:
        self.transfer_backend.close()

    def _query_weight_info(self, model_id: str) -> dict:
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        try:
            socket.connect(self.cache_endpoint)
            socket.send_pyobj({"model_id": model_id})
            return socket.recv_pyobj()
        finally:
            socket.close(linger=0)
            context.term()

    @staticmethod
    def _parse_dtype(dtype) -> torch.dtype:
        if isinstance(dtype, torch.dtype):
            return dtype
        if dtype in _DTYPE_BY_NAME:
            return _DTYPE_BY_NAME[dtype]
        raise ValueError(f"Unsupported tensor dtype from cache metadata: {dtype!r}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pull weights from a static DDR weight cache server")
    parser.add_argument("--cache-endpoint", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--bucket-size-mb", type=int, default=1024)
    parser.add_argument("--transfer-backend", default="tcp", choices=["tcp", "mooncake"])
    parser.add_argument("--transfer-protocol", default="tcp")
    parser.add_argument("--device-name", default="")
    parser.add_argument("--local-ip", default=None, help="Local IP to bind the transfer engine to")
    parser.add_argument("--recv-device", default="cpu")
    return parser


async def _main_async() -> None:
    args = build_arg_parser().parse_args()
    client = StaticWeightCacheClient(
        cache_endpoint=args.cache_endpoint,
        model_id=args.model_id,
        bucket_size=args.bucket_size_mb << 20,
        transfer_config=TransferBackendConfig(
            backend=args.transfer_backend,
            protocol=args.transfer_protocol,
            device_name=args.device_name,
        ),
        recv_device=args.recv_device,
        local_ip=args.local_ip,
    )
    try:
        count = 0
        async for _name, _tensor in client.receive_weights():
            count += 1
        logger.info("Received %s tensors", count)
    finally:
        client.close()


def main() -> None:
    import asyncio

    logging.basicConfig(level=logging.INFO)
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()


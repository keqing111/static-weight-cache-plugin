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
"""Transfer backend abstraction for static weight cache V1."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .host_mem_manager import BucketAllocation


@dataclass
class TransferBackendConfig:
    backend: str = "mooncake"
    protocol: str = "tcp"
    device_name: str = ""
    metadata_server: str = "P2PHANDSHAKE"


class TransferBackend(ABC):
    @abstractmethod
    def start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def peer_sid(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def register_buckets(self, buckets: list[BucketAllocation]) -> None:
        raise NotImplementedError

    @abstractmethod
    def read(self, peer_sid: str, dst_ptr: int, src_ptr: int, nbytes: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class MooncakeTransferBackend(TransferBackend):
    def __init__(self, local_ip: str, config: TransferBackendConfig) -> None:
        self.local_ip = local_ip
        self.config = config
        self._engine = None

    def start(self) -> None:
        from mooncake.engine import TransferEngine

        self._engine = TransferEngine()
        ret = self._engine.initialize(
            self.local_ip,
            self.config.metadata_server,
            self.config.protocol,
            self.config.device_name,
        )
        if ret != 0:
            raise RuntimeError(
                f"Failed to initialize Mooncake TransferEngine: "
                f"protocol={self.config.protocol}, ret={ret}"
            )

    def peer_sid(self) -> str:
        self._require_started()
        return f"{self.local_ip}:{self._engine.get_rpc_port()}"

    def register_buckets(self, buckets: list[BucketAllocation]) -> None:
        self._require_started()
        if not buckets:
            return
        bases = [bucket.base_ptr for bucket in buckets]
        capacities = [bucket.capacity for bucket in buckets]
        ret = self._engine.batch_register_memory(bases, capacities)
        if ret != 0:
            raise RuntimeError(f"Mooncake batch_register_memory failed: ret={ret}")

    def read(self, peer_sid: str, dst_ptr: int, src_ptr: int, nbytes: int) -> None:
        self._require_started()
        ret = self._engine.transfer_sync_read(peer_sid, dst_ptr, src_ptr, nbytes)
        if ret != 0:
            raise RuntimeError(f"Mooncake transfer_sync_read failed: ret={ret}")

    def close(self) -> None:
        self._engine = None

    def _require_started(self) -> None:
        if self._engine is None:
            raise RuntimeError("Transfer backend has not been started")


def build_transfer_backend(local_ip: str, config: TransferBackendConfig) -> TransferBackend:
    if config.backend == "mooncake":
        return MooncakeTransferBackend(local_ip=local_ip, config=config)
    raise ValueError(f"Unsupported transfer backend: {config.backend}")

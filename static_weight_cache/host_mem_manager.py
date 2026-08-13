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
"""Single-process host memory pool for static weight cache V1."""

from __future__ import annotations

import ctypes
import logging
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclass
class HostMemConfig:
    capacity_bytes: int
    alignment: int = 4096
    pin_memory: bool = True
    lock_memory: bool = False


@dataclass
class BucketAllocation:
    bucket_id: int
    offset: int
    capacity: int
    used_bytes: int
    buffer: torch.Tensor

    @property
    def base_ptr(self) -> int:
        return int(self.buffer.data_ptr())


class HostMemManager:
    """A minimal linear allocator over one long-lived host buffer.

    V1 intentionally avoids free lists, compaction, eviction and model reuse.
    The server process owns this object, so allocations survive vLLM restarts.
    """

    def __init__(self, config: HostMemConfig) -> None:
        if config.capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be positive")
        self.config = config
        self._pool_owner = torch.empty(
            config.capacity_bytes + config.alignment,
            dtype=torch.uint8,
            device="cpu",
            pin_memory=config.pin_memory,
        )
        raw_ptr = int(self._pool_owner.data_ptr())
        aligned_offset = align_up(raw_ptr, config.alignment) - raw_ptr
        self._pool = self._pool_owner.narrow(0, aligned_offset, config.capacity_bytes)
        self._next_offset = 0
        self._next_bucket_id = 0
        self._allocations: dict[int, BucketAllocation] = {}
        self._locked = False
        if config.lock_memory:
            self._lock_memory()

    @property
    def base_ptr(self) -> int:
        return int(self._pool.data_ptr())

    def allocate_bucket(self, capacity: int, used_bytes: int | None = None) -> BucketAllocation:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        capacity = align_up(capacity, self.config.alignment)
        if self._next_offset + capacity > self.config.capacity_bytes:
            raise MemoryError(
                f"Static weight cache exhausted: requested={capacity}, "
                f"free={self.config.capacity_bytes - self._next_offset}"
            )

        bucket_id = self._next_bucket_id
        self._next_bucket_id += 1
        offset = self._next_offset
        self._next_offset += capacity
        view = self._pool.narrow(0, offset, capacity)
        allocation = BucketAllocation(
            bucket_id=bucket_id,
            offset=offset,
            capacity=capacity,
            used_bytes=used_bytes if used_bytes is not None else capacity,
            buffer=view,
        )
        self._allocations[bucket_id] = allocation
        return allocation

    def get_bucket(self, bucket_id: int) -> BucketAllocation:
        return self._allocations[bucket_id]

    def write_bytes(self, allocation: BucketAllocation, offset: int, data: bytes) -> None:
        end = offset + len(data)
        if end > allocation.capacity:
            raise ValueError(f"Write exceeds bucket capacity: end={end}, capacity={allocation.capacity}")
        src = torch.frombuffer(bytearray(data), dtype=torch.uint8)
        allocation.buffer[offset:end].copy_(src)
        allocation.used_bytes = max(allocation.used_bytes, end)

    def get_registered_regions(self) -> list[BucketAllocation]:
        return list(self._allocations.values())

    def stats(self) -> dict[str, int | bool]:
        return {
            "capacity_bytes": self.config.capacity_bytes,
            "used_bytes": self._next_offset,
            "free_bytes": self.config.capacity_bytes - self._next_offset,
            "bucket_count": len(self._allocations),
            "locked": self._locked,
        }

    def close(self) -> None:
        if self._locked:
            self._unlock_memory()

    def _lock_memory(self) -> None:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        ret = libc.mlock(ctypes.c_void_p(self.base_ptr), ctypes.c_size_t(self.config.capacity_bytes))
        if ret != 0:
            errno = ctypes.get_errno()
            raise OSError(errno, "mlock failed for static weight cache pool")
        self._locked = True
        logger.info("Locked static weight cache pool: %s bytes", self.config.capacity_bytes)

    def _unlock_memory(self) -> None:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        ret = libc.munlock(ctypes.c_void_p(self.base_ptr), ctypes.c_size_t(self.config.capacity_bytes))
        if ret != 0:
            errno = ctypes.get_errno()
            logger.warning("munlock failed for static weight cache pool: errno=%s", errno)
        self._locked = False

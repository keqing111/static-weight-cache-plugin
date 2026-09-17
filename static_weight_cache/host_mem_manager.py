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
"""Single-process host memory pool for static weight cache V1.

Memory constraints on Ascend hosts
----------------------------------
Mooncake's Ascend transport is installed unconditionally by
``TransferEngine.initialize()``, and it only accepts buffers it can classify.
Plain anonymous memory -- including ``mlock``'d memory -- is reported as
``location:*`` and rejected with ``batch_register_memory`` returning -1, so the
pool must come from the Ascend pinned allocator (``pin_memory=True``).

That allocator lives in the NPU driver's devmm module, and the first
device-memory touch in a process performs a one-time ``devmm_setup_device_proc``
ioctl that blocks in uninterruptible sleep. On a busy host it has been observed
taking ~156 s, during which the process is unkillable and looks hung. Later
allocations in the same process are free. So the pool allocation is timed and
logged here rather than left silent.

``mlock`` is therefore *not* a substitute for pinning: it does not make a buffer
registrable, it only stops the pages from being swapped. It stays available for
the pageable case, where it is also the only thing it can usefully do.
"""

from __future__ import annotations

import ctypes
import logging
import time
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclass
class HostMemConfig:
    capacity_bytes: int
    alignment: int = 4096
    # Keep this True for anything the transfer engine has to register. Setting it
    # False yields a pool that cannot be registered at all on Ascend hosts.
    pin_memory: bool = True
    # Only meaningful together with pin_memory=False; pinned memory is locked by
    # the allocator already. Not a substitute for pinning.
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
        if not config.pin_memory:
            logger.warning(
                "HostMemManager built without pin_memory: the transfer engine cannot register "
                "this pool (Ascend transport reports the buffers as location:* and refuses them). "
                "Use this only for tests that never touch the transfer engine."
            )

        # A pinned allocation can block for minutes on a busy Ascend host (see
        # module docstring), so it is timed and announced rather than left
        # silent. A pageable one is a plain anonymous mapping and is not warned
        # about, because it never reaches the driver.
        if config.pin_memory:
            logger.info(
                "Allocating %.2f GB pinned host pool; the first device-memory touch in this "
                "process may block for minutes on Ascend hosts",
                config.capacity_bytes / (1 << 30),
            )
        else:
            logger.info("Allocating %.2f GB pageable host pool", config.capacity_bytes / (1 << 30))
        start = time.time()
        self._pool_owner = torch.empty(
            config.capacity_bytes + config.alignment,
            dtype=torch.uint8,
            device="cpu",
            pin_memory=config.pin_memory,
        )
        logger.info("Host pool allocated in %.2fs", time.time() - start)

        raw_ptr = int(self._pool_owner.data_ptr())
        aligned_offset = align_up(raw_ptr, config.alignment) - raw_ptr
        self._pool = self._pool_owner.narrow(0, aligned_offset, config.capacity_bytes)
        self._next_offset = 0
        self._next_bucket_id = 0
        self._allocations: dict[int, BucketAllocation] = {}
        self._locked = False
        if config.lock_memory:
            if config.pin_memory:
                logger.debug("lock_memory ignored: pinned memory is already locked by the allocator")
            else:
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
            "pin_memory": self.config.pin_memory,
            "locked": self._locked,
        }

    def close(self) -> None:
        if self._locked:
            self._unlock_memory()

    def _lock_memory(self) -> None:
        # mlock wants a page-aligned start; the pool base is aligned to
        # config.alignment, which callers must keep at the page size or larger.
        # The length is rounded up because mlock only locks whole pages.
        length = align_up(self.config.capacity_bytes, self.config.alignment)
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        ret = libc.mlock(ctypes.c_void_p(self.base_ptr), ctypes.c_size_t(length))
        if ret != 0:
            errno = ctypes.get_errno()
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
        length = align_up(self.config.capacity_bytes, self.config.alignment)
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        ret = libc.munlock(ctypes.c_void_p(self.base_ptr), ctypes.c_size_t(length))
        if ret != 0:
            errno = ctypes.get_errno()
            logger.warning("munlock failed for static weight cache pool: errno=%s", errno)
        self._locked = False

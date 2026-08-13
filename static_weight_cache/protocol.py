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
"""Wire protocol objects for static weight cache metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TensorMetadata:
    name: str
    shape: tuple[int, ...]
    dtype: str
    offset: int
    nbytes: int

    def to_wire(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["shape"] = tuple(self.shape)
        return payload


@dataclass
class BucketMetadata:
    bucket_id: int
    base_ptr: int
    capacity: int
    used_bytes: int
    tensors: dict[str, TensorMetadata] = field(default_factory=dict)

    def to_flexfetch_bucket_meta(self) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "shape": tensor.shape,
                "dtype": tensor.dtype,
                "offset": tensor.offset,
                "nbytes": tensor.nbytes,
            }
            for name, tensor in self.tensors.items()
        }


@dataclass
class ModelMetadata:
    model_id: str
    revision: str | None
    bucket_num: int
    buckets: list[BucketMetadata]
    status: str = "READY"

    def to_weight_info(self) -> dict[str, Any]:
        return {
            "bucket_num": self.bucket_num,
            "bucket_meta": [bucket.to_flexfetch_bucket_meta() for bucket in self.buckets],
            "bases": [bucket.base_ptr for bucket in self.buckets],
            "capacities": [bucket.capacity for bucket in self.buckets],
            "used_bytes": [bucket.used_bytes for bucket in self.buckets],
        }


@dataclass
class CacheQuery:
    model_id: str
    revision: str | None = None

    @classmethod
    def from_wire(cls, payload: Any) -> "CacheQuery":
        if isinstance(payload, str):
            return cls(model_id=payload)
        if isinstance(payload, dict):
            return cls(model_id=payload["model_id"], revision=payload.get("revision"))
        raise TypeError(f"Unsupported cache query payload: {type(payload)!r}")


@dataclass
class CacheResponse:
    ok: bool
    peer_sid: str | None = None
    weight_info: dict[str, Any] | None = None
    error: str | None = None

    def to_wire(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "peer_sid": self.peer_sid,
            "weight_info": self.weight_info or {},
            "error": self.error,
        }

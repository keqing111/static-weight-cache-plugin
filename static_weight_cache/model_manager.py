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
"""Model-to-bucket loading for static weight cache V1."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import torch

from .host_mem_manager import BucketAllocation, HostMemManager
from .protocol import BucketMetadata, ModelMetadata, TensorMetadata

logger = logging.getLogger(__name__)


class ModelManager:
    """Loads model weights into HostMemManager and owns model metadata."""

    def __init__(self, host_mem_manager: HostMemManager, bucket_size_bytes: int) -> None:
        if bucket_size_bytes <= 0:
            raise ValueError("bucket_size_bytes must be positive")
        self.host_mem_manager = host_mem_manager
        self.bucket_size_bytes = bucket_size_bytes
        self._models: dict[str, ModelMetadata] = {}

    def list_models(self) -> list[str]:
        return list(self._models.keys())

    def get_model(self, model_id: str) -> ModelMetadata:
        return self._models[model_id]

    def load_from_named_tensors(
        self,
        model_id: str,
        named_tensors: Iterable[tuple[str, torch.Tensor]],
        revision: str | None = None,
    ) -> ModelMetadata:
        if model_id in self._models:
            raise ValueError(f"Model already loaded: {model_id}")

        buckets: list[BucketMetadata] = []
        current_allocation: BucketAllocation | None = None
        current_tensors: dict[str, TensorMetadata] = {}
        current_offset = 0

        def flush_bucket() -> None:
            nonlocal current_allocation, current_tensors, current_offset
            if current_allocation is None:
                return
            current_allocation.used_bytes = current_offset
            buckets.append(
                BucketMetadata(
                    bucket_id=current_allocation.bucket_id,
                    base_ptr=current_allocation.base_ptr,
                    capacity=current_allocation.capacity,
                    used_bytes=current_allocation.used_bytes,
                    tensors=current_tensors,
                )
            )
            current_allocation = None
            current_tensors = {}
            current_offset = 0

        for name, tensor in named_tensors:
            tensor = tensor.detach().cpu().contiguous()
            nbytes = int(tensor.numel() * tensor.element_size())
            if nbytes > self.bucket_size_bytes:
                flush_bucket()
                current_allocation = self.host_mem_manager.allocate_bucket(nbytes, used_bytes=0)
                written = self._copy_tensor_to_bucket(current_allocation, 0, tensor)
                current_offset = written
                current_tensors[name] = TensorMetadata(
                    name=name,
                    shape=tuple(tensor.shape),
                    dtype=str(tensor.dtype),
                    offset=0,
                    nbytes=written,
                )
                flush_bucket()
                continue

            if current_allocation is None:
                current_allocation = self.host_mem_manager.allocate_bucket(self.bucket_size_bytes, used_bytes=0)
            elif current_offset + nbytes > current_allocation.capacity:
                flush_bucket()
                current_allocation = self.host_mem_manager.allocate_bucket(self.bucket_size_bytes, used_bytes=0)

            tensor_offset = current_offset
            written = self._copy_tensor_to_bucket(current_allocation, tensor_offset, tensor)
            current_offset += written
            current_tensors[name] = TensorMetadata(
                name=name,
                shape=tuple(tensor.shape),
                dtype=str(tensor.dtype),
                offset=tensor_offset,
                nbytes=written,
            )

        flush_bucket()
        metadata = ModelMetadata(
            model_id=model_id,
            revision=revision,
            bucket_num=len(buckets),
            buckets=buckets,
        )
        self._models[model_id] = metadata
        logger.info("Loaded model %s into %s buckets", model_id, metadata.bucket_num)
        return metadata

    def load_safetensors_dir(
        self,
        model_id: str,
        model_path: str | Path,
        revision: str | None = None,
    ) -> ModelMetadata:
        try:
            from safetensors.torch import safe_open
        except ImportError as err:
            raise RuntimeError("safetensors is required to load safetensors model files") from err

        root = Path(model_path)
        files = sorted(root.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"No safetensors files found under {root}")

        def iter_tensors() -> Iterable[tuple[str, torch.Tensor]]:
            for file_path in files:
                with safe_open(file_path, framework="pt", device="cpu") as handle:
                    for key in handle.keys():
                        yield key, handle.get_tensor(key)

        return self.load_from_named_tensors(model_id=model_id, named_tensors=iter_tensors(), revision=revision)

    @staticmethod
    def _copy_tensor_to_bucket(allocation: BucketAllocation, offset: int, tensor: torch.Tensor) -> int:
        src = tensor.view(-1).view(torch.uint8)
        end = offset + int(src.numel())
        if end > allocation.capacity:
            raise ValueError(f"Write exceeds bucket capacity: end={end}, capacity={allocation.capacity}")
        allocation.buffer[offset:end].copy_(src)
        allocation.used_bytes = max(allocation.used_bytes, end)
        return int(src.numel())

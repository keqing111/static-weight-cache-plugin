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
"""CheckpointEngine backend wrapper for the static weight cache client."""

from __future__ import annotations

from .client import StaticWeightCacheClient
from .transfer_backend import TransferBackendConfig

try:
    from verl.checkpoint_engine.base import CheckpointEngine, CheckpointEngineRegistry
except ImportError:  # pragma: no cover - useful when running lightweight tooling.
    CheckpointEngine = object
    CheckpointEngineRegistry = None


class _StaticWeightCacheCheckpointEngine(CheckpointEngine):
    """Minimal CheckpointEngine-compatible receiver.

    V1 intentionally leaves process-group metadata and rank0 broadcast out of
    this class. It is suitable for single-rank smoke tests and gives the vLLM
    loader the same async generator shape as existing CheckpointEngine backends.
    """

    def __init__(
        self,
        bucket_size: int,
        cache_endpoint: str,
        model_id: str,
        transfer_backend: str = "mooncake",
        transfer_protocol: str = "tcp",
        device_name: str = "",
        recv_device: str = "cpu",
        **_kwargs,
    ) -> None:
        self.client = StaticWeightCacheClient(
            cache_endpoint=cache_endpoint,
            model_id=model_id,
            bucket_size=bucket_size,
            recv_device=recv_device,
            transfer_config=TransferBackendConfig(
                backend=transfer_backend,
                protocol=transfer_protocol,
                device_name=device_name,
            ),
        )

    def prepare(self):
        return None

    def init_process_group(self, *args, **kwargs) -> None:
        return None

    async def receive_weights(self, **kwargs):
        model_id = kwargs.get("model_id")
        async for item in self.client.receive_weights(model_id=model_id):
            yield item

    async def send_weights(self, *_args, **_kwargs):
        raise NotImplementedError("static_weight_cache is a receive-only backend")

    def finalize(self) -> None:
        return None

    def close(self) -> None:
        self.client.close()


if CheckpointEngineRegistry is not None:
    StaticWeightCacheCheckpointEngine = CheckpointEngineRegistry.register("static_weight_cache")(
        _StaticWeightCacheCheckpointEngine
    )
else:
    StaticWeightCacheCheckpointEngine = _StaticWeightCacheCheckpointEngine

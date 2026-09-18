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
"""把模型权重装进内存池，并产出描述它的元数据。

本模块是"权重从哪来"和"权重怎么摆放"之间的那一层。它从 safetensors（或任何
``(name, tensor)`` 形式的可迭代对象）读入 tensor，按顺序贪心打包进固定容量的
bucket，同时记录每个 tensor 在 bucket 内的字节偏移。

打包结果决定了两个下游契约：

1. **传输粒度是 bucket** —— 客户端一次读一整块连续内存，而不是逐个 tensor。
2. **元数据只描述相对偏移** —— 客户端拿到 bucket 数据后，按 offset/shape/dtype
   在本地切出各个 tensor。

所以"哪个 tensor 落在哪个 bucket"完全由这里的顺序和容量规则决定，见
``load_from_named_tensors`` 的注释。
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import torch

from .host_mem_manager import BucketAllocation, HostMemManager
from .protocol import BucketMetadata, ModelMetadata, TensorMetadata

logger = logging.getLogger(__name__)


class ModelManager:
    """把模型权重加载进 HostMemManager，并持有其元数据。"""

    def __init__(self, host_mem_manager: HostMemManager, bucket_size_bytes: int) -> None:
        if bucket_size_bytes <= 0:
            raise ValueError("bucket_size_bytes must be positive")
        # 池子由外部持有：服务进程常驻，池子也要跟着常驻，不能随本对象生命周期走。
        self.host_mem_manager = host_mem_manager
        # 常规 bucket 的容量上限。注意这是"上限"不是"固定大小"：装不下才切新的，
        # 所以末尾那个 bucket 通常只用了其中一部分。
        self.bucket_size_bytes = bucket_size_bytes
        # model_id -> ModelMetadata。已加载的模型都记在这里，查询接口直接按 id 取。
        self._models: dict[str, ModelMetadata] = {}

    def list_models(self) -> list[str]:
        """已加载的 model_id 列表。"""
        return list(self._models.keys())

    def get_model(self, model_id: str) -> ModelMetadata:
        """按 model_id 取元数据；不存在会抛 KeyError，由服务端转成错误响应。"""
        return self._models[model_id]

    def load_from_named_tensors(
        self,
        model_id: str,
        named_tensors: Iterable[tuple[str, torch.Tensor]],
        revision: str | None = None,
    ) -> ModelMetadata:
        """把 ``(name, tensor)`` 流打包进 bucket，并登记元数据。

        打包规则（这是本文件的核心，也是"哪个 tensor 进哪个 bucket"的唯一答案）::

            for 每个 tensor（按调用方给出的顺序）:
                nbytes = numel * element_size
                if nbytes > bucket_size_bytes：      # 单个 tensor 就超过一块 bucket
                    冲掉当前 bucket，为它单独开一块刚好装得下的 bucket，写完立刻冲掉
                elif 当前 bucket 装不下：
                    冲掉当前 bucket，开一块新的
                追加写入当前 bucket，记录它在 bucket 内的偏移

        **顺序由调用方决定，本函数不做任何排序。** ``load_safetensors_dir`` 传进来的
        顺序是：文件名排序 → 文件内 key 顺序。所以 bucket 边界完全由"贪心装满就切"
        决定，和 safetensors 的文件边界没有关系：一个 bucket 可以横跨两个文件，
        一个文件也可以被切成多个 bucket。

        另外注意这里没有做形状/dtype 的转换，直接按原样拷贝字节。相比之下
        flexfetch 会先按 rollout_dtype 预算好大小（``deterministic_nbytes``），
        从而保证同一个 layer 在不同版本间落在相同偏移、可以复用地址；本实现没有
        这个性质。
        """
        if model_id in self._models:
            # V1 没有淘汰与复用机制，池子是线性分配的，重复加载会白白多占一份内存。
            raise ValueError(f"Model already loaded: {model_id}")

        buckets: list[BucketMetadata] = []
        current_allocation: BucketAllocation | None = None
        current_tensors: dict[str, TensorMetadata] = {}
        current_offset = 0

        def flush_bucket() -> None:
            """收尾当前 bucket：回填 used_bytes，生成元数据，然后重置状态。

            用闭包改外层变量，是为了让主循环里的"切桶"逻辑保持短小。
            """
            nonlocal current_allocation, current_tensors, current_offset
            if current_allocation is None:
                return
            # 主循环里分配时 used_bytes 填的是 0，到这里才知道真正用了多少。
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
            # detach：权重只读，不需要梯度；cpu：池子在 host 上；
            # contiguous：后面是按字节区间连续拷贝的，非连续布局会把数据写错。
            tensor = tensor.detach().cpu().contiguous()
            nbytes = int(tensor.numel() * tensor.element_size())

            # 单独处理超大 tensor：它自己就超过一块常规 bucket，塞不进任何一组。
            # 为它开一块刚好等于它大小的 bucket，写完立即冲掉，避免它把后面的
            # tensor 都挤到下一块去。
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

            # 常规路径：没有在用中的 bucket 就开一块，装不下就切一块新的。
            if current_allocation is None:
                current_allocation = self.host_mem_manager.allocate_bucket(self.bucket_size_bytes, used_bytes=0)
            elif current_offset + nbytes > current_allocation.capacity:
                flush_bucket()
                current_allocation = self.host_mem_manager.allocate_bucket(self.bucket_size_bytes, used_bytes=0)

            # 记下这个 tensor 在 bucket 内的相对偏移 —— 客户端就是靠它从一整块
            # bucket 数据里切出各个 tensor 的。
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

        # 循环结束后还有没冲掉的最后一块。
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
        """从一个目录里的 safetensors 文件加载模型。

        读取顺序 = ``sorted(glob("*.safetensors"))`` 的文件名字典序，文件内按
        ``handle.keys()`` 顺序（即 safetensors 头部里记录的顺序）。这个顺序会原样
        传给 ``load_from_named_tensors``，进而决定 bucket 的切分位置。

        注意本函数不解析 ``model.safetensors.index.json``，而是无差别地读入目录下
        所有 safetensors 的全部 key —— 包括训练时不会用到的那些。
        """
        try:
            from safetensors.torch import safe_open
        except ImportError as err:
            raise RuntimeError("safetensors is required to load safetensors model files") from err

        root = Path(model_path)
        files = sorted(root.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"No safetensors files found under {root}")

        def iter_tensors() -> Iterable[tuple[str, torch.Tensor]]:
            """惰性生成 (name, tensor)。

            用生成器而不是先全部读进内存，是因为整个模型有十几 GB；安全张量库内部
            是 mmap，逐 key 取用时才会真正触碰对应页。顺带这也让"从共享盘读"这件事
            按需发生，与下面的打包循环天然流水起来。
            """
            for file_path in files:
                with safe_open(file_path, framework="pt", device="cpu") as handle:
                    for key in handle.keys():
                        yield key, handle.get_tensor(key)

        return self.load_from_named_tensors(model_id=model_id, named_tensors=iter_tensors(), revision=revision)

    @staticmethod
    def _copy_tensor_to_bucket(allocation: BucketAllocation, offset: int, tensor: torch.Tensor) -> int:
        """把 tensor 的原始字节拷进 bucket 的指定偏移，返回写入的字节数。

        先 ``view(-1)`` 拉平再 ``view(torch.uint8)``，是为了把任意 dtype 统一当成
        字节流处理 —— 池子里存的就是原始字节，不含 dtype 信息，dtype 由元数据描述。
        拷贝前会再检查一次边界，避免越界写坏相邻 bucket。
        """
        src = tensor.view(-1).view(torch.uint8)
        end = offset + int(src.numel())
        if end > allocation.capacity:
            raise ValueError(f"Write exceeds bucket capacity: end={end}, capacity={allocation.capacity}")
        allocation.buffer[offset:end].copy_(src)
        allocation.used_bytes = max(allocation.used_bytes, end)
        return int(src.numel())

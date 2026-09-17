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
"""端到端验证：vLLM 经过缓存插件加载权重，与直接从共享盘加载对比。

为什么必须走 vLLM 而不是单纯 server→client 搬运
----------------------------------------------
搬运本身不是收益。收益是"vLLM 的权重加载变快"，所以两次测量都必须经过
vLLM 真实的那条加载路径，也就是 ``nn.Module.load_weights(Iterable[(name, tensor)])``
（`vllm/model_executor/models/qwen3.py:338`）。插件产出的正是一个
``(name, cpu_tensor)`` 流，所以两者可以直接替换、其余代码完全一致。

对比的两条路径
--------------
  disk   : get_model(vllm_config)               —— vLLM 默认 loader，从 NFS 读 safetensors
  plugin : model.load_weights(client.receive_weights())  —— 从缓存服务进程拉

两次加载结束后对全部参数做一次全量摘要对比，确认插件路径拿到的权重逐字节一致
（只看耗时而不校验，等于没验证）。

用法
----
    python3 bench/e2e_vllm_load.py --model-path /data/y50063564/qwen3-8B-dfly/qwen3-8B
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29511")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("VLLM_ASCEND_ENABLE_NZ", "0")

import torch  # noqa: E402
import torch_npu  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def stamp(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------
# 缓存服务进程
# --------------------------------------------------------------------------


class CacheServerProcess:
    """把缓存服务跑到独立进程里，和真实部署一致（独立地址空间）。"""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.proc: subprocess.Popen | None = None

    def start(self) -> None:
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "static_weight_cache.server",
            "--model-id",
            self.args.model_id,
            "--model-path",
            self.args.model_path,
            "--bind-ip",
            self.args.local_ip,
            "--metadata-port",
            str(self.args.metadata_port),
            "--capacity-gb",
            str(self.args.pool_gb),
            "--bucket-size-mb",
            str(self.args.bucket_mb),
            "--transfer-backend",
            "tcp",
        ]
        stamp(f"starting cache server: capacity={self.args.pool_gb}GB bucket={self.args.bucket_mb}MB")
        # Log to a file rather than a pipe: nobody drains a pipe while we poll for
        # readiness, so a chatty server would eventually block on a full buffer.
        self.log_path = Path(f"/tmp/e2e_cache_server_{self.args.metadata_port}.log")
        self.log_file = self.log_path.open("w")
        self.proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self._wait_ready()

    def _wait_ready(self, timeout: float = 3600.0) -> None:
        """服务要把整个模型读进池子才可用，所以这里的超时给得很宽。

        池子是 pinned 的，在这台机器上意味着要走 NPU 驱动的 devmm 路径，首次
        触摸设备内存可能阻塞数分钟；再加上从共享盘单线程读 16 GB，整体很慢是正常的。
        """
        deadline = time.time() + timeout
        last_report = time.time()
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"cache server exited early; see {self.log_path}")
            if self._probe():
                stamp("cache server is serving")
                return
            if time.time() - last_report > 60:
                last_report = time.time()
                stamp(f"still waiting for cache server ({self.log_path})")
            time.sleep(2.0)
        raise TimeoutError("cache server did not become ready")

    def _probe(self) -> bool:
        import zmq

        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, 1000)
        socket.setsockopt(zmq.SNDTIMEO, 1000)
        try:
            socket.connect(f"tcp://{self.args.local_ip}:{self.args.metadata_port}")
            socket.send_pyobj({"model_id": self.args.model_id})
            return bool(socket.recv_pyobj().get("ok"))
        except Exception:  # noqa: BLE001 - not ready yet
            return False
        finally:
            socket.close(linger=0)
            context.term()

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if getattr(self, "log_file", None):
            self.log_file.close()


# --------------------------------------------------------------------------
# 把异步生成器接到同步的 load_weights 上
# --------------------------------------------------------------------------


def sync_iter(async_gen):
    """vLLM 的 load_weights 要同步可迭代对象，插件的接口是异步生成器。

    客户端内部是阻塞式读，所以这里用一个私有事件循环逐步驱动即可，不改变
    实际执行顺序。
    """
    loop = asyncio.new_event_loop()
    iterator = async_gen.__aiter__()
    try:
        while True:
            try:
                yield loop.run_until_complete(iterator.__anext__())
            except StopAsyncIteration:
                return
    finally:
        loop.close()


# --------------------------------------------------------------------------
# 参数校验
# --------------------------------------------------------------------------


def digest_params(model) -> dict[str, str]:
    """逐参数算摘要，返回 {参数名: sha256}。

    做成逐参数而不是一个总摘要，是为了在对不上时能直接指出是哪些参数错了——
    总摘要只能告诉你"不一样"，没法区分是插件传错了数据，还是某些参数本来就不在
    检查点里。逐参数搬到 host 再算，峰值只占一个参数的大小。
    """
    digests: dict[str, str] = {}
    with torch.no_grad():
        for name, param in sorted(model.named_parameters()):
            raw = param.detach().to("cpu").contiguous().view(torch.uint8).numpy().tobytes()
            digests[name] = hashlib.sha256(raw).hexdigest()
    stamp(f"digested {len(digests)} parameters")
    return digests


def compare_digests(expected: dict[str, str], actual: dict[str, str], limit: int = 10) -> list[str]:
    """返回不一致的参数名；只报告不看内容，避免把 16GB 的权重打出来。"""
    names = sorted(set(expected) | set(actual))
    return [name for name in names if expected.get(name) != actual.get(name)][:limit]


def zero_params(model) -> None:
    """把参数清零，好让下一次加载真的从零开始，而不是叠加上一次的结果。"""
    with torch.no_grad():
        for param in model.parameters():
            param.data.zero_()


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------


def build_vllm_config(args: argparse.Namespace):
    from vllm.engine.arg_utils import EngineArgs

    engine_args = EngineArgs(
        model=args.model_path,
        tensor_parallel_size=1,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    return engine_args.create_engine_config()


def main() -> int:
    parser = argparse.ArgumentParser(description="vLLM 权重加载：共享盘 vs 缓存插件")
    parser.add_argument("--model-path", default="/data/y50063564/qwen3-8B-dfly/qwen3-8B")
    parser.add_argument("--model-id", default="qwen3-8b")
    parser.add_argument("--local-ip", default="172.26.4.186")
    parser.add_argument("--metadata-port", type=int, default=5599)
    parser.add_argument("--pool-gb", type=float, default=20.0)
    parser.add_argument("--bucket-mb", type=int, default=512)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--skip-disk", action="store_true", help="只测插件路径，跳过 NFS 基线")
    args = parser.parse_args()

    torch.npu.set_device(args.device_id)

    from vllm.config import set_current_vllm_config
    from vllm.model_executor.model_loader import get_model
    from vllm.v1.worker.gpu_worker import init_worker_distributed_environment

    vllm_config = build_vllm_config(args)

    # 缓存服务先把模型从共享盘读进 host 内存池，这一步不计入 vLLM 加载耗时：
    # 它是"一次性的预热成本"，正是缓存要摊掉的东西。
    server = CacheServerProcess(args)
    server.start()
    try:
        stamp("building vLLM model (weights come from the default loader)")
        with set_current_vllm_config(vllm_config):
            init_worker_distributed_environment(vllm_config, rank=0, local_rank=0, backend="hccl")

        t0 = time.perf_counter()
        with set_current_vllm_config(vllm_config):
            model = get_model(vllm_config=vllm_config)
        model.eval()
        disk_seconds = time.perf_counter() - t0
        stamp(f"disk load finished in {disk_seconds:.2f}s")

        disk_digest = None
        if not args.skip_disk:
            disk_digest = digest_params(model)

        # ---- 插件路径：同一个模型实例，参数清零后从缓存重新加载 ----
        from static_weight_cache.client import StaticWeightCacheClient
        from static_weight_cache.transfer_backend import TransferBackendConfig

        zero_params(model)
        if disk_digest is not None:
            # 先确认摘要本身是敏感的：清零必须让每一个参数都变。否则后面
            # "两边一致"可能只是因为这个校验根本检测不出变化。
            zeroed = digest_params(model)
            unchanged = [name for name, dig in zeroed.items() if disk_digest.get(name) == dig]
            if unchanged:
                raise AssertionError(
                    f"digest is not sensitive: {len(unchanged)} parameters unchanged after zeroing, "
                    f"e.g. {unchanged[:5]}"
                )
            stamp("sanity check OK: zeroing changed every parameter digest")

        stamp("loading through the cache plugin")

        client = StaticWeightCacheClient(
            cache_endpoint=f"tcp://{args.local_ip}:{args.metadata_port}",
            model_id=args.model_id,
            bucket_size=args.bucket_mb << 20,
            transfer_config=TransferBackendConfig(backend="tcp"),
            local_ip=args.local_ip,
        )
        try:
            t0 = time.perf_counter()
            loaded = model.load_weights(sync_iter(client.receive_weights()))
            plugin_seconds = time.perf_counter() - t0
            stamp(f"plugin load finished in {plugin_seconds:.2f}s, {len(loaded)} weights")
        finally:
            client.close()

        plugin_digest = digest_params(model)

        print("\n" + "=" * 66)
        if disk_digest is not None:
            print(f"  disk   (NFS -> vLLM)   : {disk_seconds:8.2f} s")
        print(f"  plugin (cache -> vLLM) : {plugin_seconds:8.2f} s")

        ok = True
        if disk_digest is not None:
            mismatched = compare_digests(disk_digest, plugin_digest)
            ok = not mismatched
            print(f"  speedup                : {disk_seconds / plugin_seconds:8.2f} x")
            print(f"  weights identical      : {'YES' if ok else 'NO'}")
            if mismatched:
                print(f"  first mismatching params: {mismatched}")
        print("=" * 66)
        return 0 if ok else 1
    finally:
        server.stop()


if __name__ == "__main__":
    raise SystemExit(main())

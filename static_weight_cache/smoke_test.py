"""Segmented smoke tests for the standalone static weight cache plugin."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from collections.abc import Iterable

import torch
import zmq

from .client import StaticWeightCacheClient
from .host_mem_manager import HostMemConfig, HostMemManager
from .model_manager import ModelManager
from .server import WeightCacheServer, WeightCacheServerConfig
from .transfer_backend import TransferBackendConfig, build_transfer_backend

logger = logging.getLogger(__name__)
MODEL_ID = "static-weight-cache-smoke"


def make_test_tensors() -> list[tuple[str, torch.Tensor]]:
    return [
        ("embed.weight", torch.arange(128, dtype=torch.float32).reshape(16, 8)),
        ("layer.0.weight", torch.arange(96, dtype=torch.float16).reshape(12, 8)),
        ("layer.0.bias", torch.arange(12, dtype=torch.int32)),
        ("lm_head.weight", torch.arange(64, dtype=torch.uint8).reshape(8, 8)),
    ]


def clone_named_tensors(named_tensors: Iterable[tuple[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in named_tensors}


def print_stage(name: str, ok: bool, detail: str = "") -> None:
    suffix = f" - {detail}" if detail else ""
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{suffix}", flush=True)


def query_metadata(endpoint: str, model_id: str, timeout_ms: int) -> dict:
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
    try:
        socket.connect(endpoint)
        socket.send_pyobj({"model_id": model_id})
        return socket.recv_pyobj()
    finally:
        socket.close(linger=0)
        context.term()


def compare_tensors(expected: dict[str, torch.Tensor], actual: dict[str, torch.Tensor]) -> None:
    if set(expected) != set(actual):
        raise AssertionError(f"tensor names differ: expected={sorted(expected)}, actual={sorted(actual)}")
    for name, expected_tensor in expected.items():
        actual_tensor = actual[name]
        if expected_tensor.shape != actual_tensor.shape:
            raise AssertionError(f"shape mismatch for {name}: {expected_tensor.shape} != {actual_tensor.shape}")
        if expected_tensor.dtype != actual_tensor.dtype:
            raise AssertionError(f"dtype mismatch for {name}: {expected_tensor.dtype} != {actual_tensor.dtype}")
        if not torch.equal(expected_tensor, actual_tensor):
            raise AssertionError(f"value mismatch for {name}")


async def collect_client_weights(client: StaticWeightCacheClient) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    async for name, tensor in client.receive_weights():
        result[name] = tensor.detach().cpu().clone()
    return result


def make_transfer_config(args: argparse.Namespace) -> TransferBackendConfig:
    return TransferBackendConfig(
        backend=args.transfer_backend,
        protocol=args.transfer_protocol,
        device_name=args.device_name,
    )


def resolve_pin_memory(args: argparse.Namespace) -> bool:
    """Pinned by default: the pool is the cache and must not be reclaimable."""
    return not args.no_pin_memory


def make_server(args: argparse.Namespace) -> WeightCacheServer:
    pin_memory = resolve_pin_memory(args)
    return WeightCacheServer(
        WeightCacheServerConfig(
            model_id=MODEL_ID,
            bind_ip=args.bind_ip,
            metadata_port=args.metadata_port,
            capacity_bytes=args.capacity_mb << 20,
            bucket_size_bytes=args.bucket_size_kb << 10,
            host_mem=HostMemConfig(
                capacity_bytes=args.capacity_mb << 20,
                pin_memory=pin_memory,
                lock_memory=args.lock_memory,
            ),
            transfer=make_transfer_config(args),
        )
    )


def smoke_imports(_args: argparse.Namespace) -> None:
    # mooncake is only needed by the mooncake backend, so it is reported rather
    # than required; the default TCP backend does not import it.
    try:
        import mooncake.engine  # noqa: F401

        have_mooncake = "mooncake"
    except ImportError:
        have_mooncake = "no mooncake"
    print_stage("import dependencies", True, f"torch/zmq/{have_mooncake}")


def smoke_memory(args: argparse.Namespace) -> None:
    named_tensors = make_test_tensors()
    mem = HostMemManager(
        HostMemConfig(
            capacity_bytes=args.capacity_mb << 20,
            pin_memory=resolve_pin_memory(args),
            lock_memory=args.lock_memory,
        )
    )
    try:
        manager = ModelManager(mem, args.bucket_size_kb << 10)
        metadata = manager.load_from_named_tensors(MODEL_ID, named_tensors)
        stats = mem.stats()
        if metadata.bucket_num <= 0:
            raise AssertionError("no buckets were created")
        print_stage("host memory + model bucketization", True, f"buckets={metadata.bucket_num} used={stats['used_bytes']}")
    finally:
        mem.close()


def smoke_backend_init(args: argparse.Namespace) -> None:
    backend = build_transfer_backend(args.bind_ip, make_transfer_config(args))
    try:
        backend.start()
        print_stage(f"{args.transfer_backend} transfer backend init", True, backend.peer_sid())
    finally:
        backend.close()


def smoke_metadata(args: argparse.Namespace) -> str:
    server = make_server(args)
    try:
        server.start()
        server.load_named_tensors(MODEL_ID, make_test_tensors())
        response = query_metadata(server.metadata_endpoint, MODEL_ID, args.query_timeout_ms)
        if not response.get("ok"):
            raise RuntimeError(response.get("error") or "metadata query returned ok=False")
        info = response["weight_info"]
        if info["bucket_num"] <= 0:
            raise AssertionError("metadata contains no buckets")
        print_stage("ZMQ metadata query", True, f"endpoint={server.metadata_endpoint} buckets={info['bucket_num']}")
        return server.metadata_endpoint
    finally:
        server.stop()


async def smoke_transfer(args: argparse.Namespace) -> None:
    named_tensors = make_test_tensors()
    expected = clone_named_tensors(named_tensors)
    server = make_server(args)
    client = None
    start = time.time()
    try:
        server.start()
        server.load_named_tensors(MODEL_ID, named_tensors)
        client = StaticWeightCacheClient(
            cache_endpoint=server.metadata_endpoint,
            model_id=MODEL_ID,
            bucket_size=args.bucket_size_kb << 10,
            transfer_config=make_transfer_config(args),
            recv_device="cpu",
            # Without this the client follows the default route and can end up
            # on a different NIC than the server, which the engine then cannot
            # reach across the Ascend link.
            local_ip=args.bind_ip,
        )
        actual = await collect_client_weights(client)
        compare_tensors(expected, actual)
        print_stage("TransferEngine read + receive_weights", True, f"tensors={len(actual)} elapsed={time.time() - start:.3f}s")
    finally:
        if client is not None:
            client.close()
        server.stop()


async def run_smoke(args: argparse.Namespace) -> None:
    if args.case in ("imports", "all"):
        smoke_imports(args)
    if args.case in ("memory", "all"):
        smoke_memory(args)
    if args.case in ("backend-init", "mooncake-init", "all"):
        smoke_backend_init(args)
    if args.case in ("metadata", "all"):
        smoke_metadata(args)
    if args.case in ("transfer", "all"):
        await smoke_transfer(args)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run segmented static weight cache smoke tests")
    parser.add_argument("--case", choices=["imports", "memory", "backend-init", "metadata", "transfer", "all"], default="all")
    parser.add_argument("--bind-ip", default="127.0.0.1")
    parser.add_argument("--metadata-port", type=int, default=0)
    parser.add_argument("--capacity-mb", type=int, default=64)
    parser.add_argument("--bucket-size-kb", type=int, default=64)
    parser.add_argument("--transfer-backend", default="tcp", choices=["tcp", "mooncake"])
    parser.add_argument("--transfer-protocol", default="tcp")
    parser.add_argument("--device-name", default="")
    # Both default to unset, so the pool follows the transfer backend: pinned is
    # only needed by mooncake, whose Ascend transport rejects anything it cannot
    # classify. Override when you want to exercise a specific pool type.
    parser.add_argument(
        "--no-pin-memory",
        action="store_true",
        help="pageable pool; the cache then becomes reclaimable and the pool is no longer RDMA-registrable",
    )
    parser.add_argument("--lock-memory", action="store_true")
    parser.add_argument("--query-timeout-ms", type=int, default=5000)
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    args = build_arg_parser().parse_args()
    try:
        asyncio.run(run_smoke(args))
    except Exception as err:
        print_stage(f"smoke case={args.case}", False, repr(err))
        return 1
    print_stage(f"smoke case={args.case}", True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

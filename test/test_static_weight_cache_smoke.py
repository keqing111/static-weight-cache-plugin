# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import argparse
import asyncio

import pytest


def test_static_weight_cache_mooncake_smoke():
    pytest.importorskip("mooncake.engine")

    from static_weight_cache.smoke_test import run_smoke

    args = argparse.Namespace(
        bind_ip="127.0.0.1",
        metadata_port=0,
        capacity_mb=64,
        bucket_size_kb=64,
        transfer_backend="mooncake",
        transfer_protocol="tcp",
        device_name="",
        pin_memory=False,
        lock_memory=False,
        query_timeout_ms=5000,
    )
    asyncio.run(run_smoke(args))

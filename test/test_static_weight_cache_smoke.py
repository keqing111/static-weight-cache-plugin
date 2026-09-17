# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import argparse
import asyncio

from static_weight_cache.smoke_test import run_smoke


def test_static_weight_cache_smoke():
    """Exercise every smoke stage against the default TCP transfer backend.

    The TCP backend talks to no vendor library, so this runs anywhere. The
    mooncake backend is covered by running the smoke module directly with
    ``--transfer-backend mooncake`` on a host where that path is healthy.
    """
    args = argparse.Namespace(
        case="all",
        bind_ip="127.0.0.1",
        metadata_port=0,
        capacity_mb=64,
        bucket_size_kb=64,
        transfer_backend="tcp",
        transfer_protocol="tcp",
        device_name="",
        no_pin_memory=False,
        lock_memory=False,
        query_timeout_ms=5000,
    )
    asyncio.run(run_smoke(args))

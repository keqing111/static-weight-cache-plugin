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
"""共享盘（NFS）读取带宽扫描 —— 多流拉取的上限测量。

静态权重缓存的冷路径只发生一次：把模型从共享盘搬进内存池。这条路径的上限，
就是"能开多少条并发流、聚合带宽能到多少"。本工具扫描流数，输出聚合带宽和
单次 pread 的时延分布。

为什么要用 O_DIRECT
-------------------
本机有 2 TB 内存，读一遍 16 GB 的模型就全部进了 page cache，之后再读量到的是
本地内存带宽而不是 NFS 服务端。用 O_DIRECT 打开文件后，page cache 既不参与
服务、也不吸收数据：每一次 pread 在链路上都是一次真实的 NFS READ。这样反复
跑多遍结果依然可信，而且不需要用 /proc/sys/vm/drop_caches 去动全局 page cache
（那会影响同机其他用户）。

为什么缓冲区用 mmap 而不是 pinned
----------------------------------
目标缓冲区是否 pin 住，不影响网络侧的带宽，只影响后面有没有一次额外拷贝。
而在这台机器上 torch.empty(pin_memory=True) 会走进 NPU 驱动的 devmm 路径并
无限期阻塞在不可中断睡眠里（实测内核栈停在 drv_devmm_host），所以测量缓冲区
一律用 mmap.mmap(-1, ...)，它天然页对齐且是普通匿名页。想把"拷进 pinned"这
一步的代价单独量出来，用 --pin-dest。

为什么用线程就够
----------------
os.preadv 在内核走 RPC 期间会释放 GIL，所以限制吞吐的是流数而不是解释器。
--mode process 存在的意义是用实测去证实这一点，而不是想当然。

用法
----
    # 带宽扫描：整个模型，O_DIRECT，多线程
    python3 bench/nfs_scan.py --model-path /data/y50063564/qwen3-8B-dfly/qwen3-8B

    # 换成多进程跑，验证 GIL 不是瓶颈
    python3 bench/nfs_scan.py --model-path ... --mode process --streams 8,12

    # 额外把每块数据拷进 pinned，量这次拷贝的代价（本机慎用，见上）
    python3 bench/nfs_scan.py --model-path ... --streams 12 --pin-dest

    # 对照组：不用 O_DIRECT，量 page cache 命中时的内存带宽
    python3 bench/nfs_scan.py --model-path ... --streams 12 --no-direct
"""

from __future__ import annotations

import argparse
import json
import mmap
import multiprocessing
import os
import statistics
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

# O_DIRECT 以及它的读操作要求的缓冲区对齐。取 4 KB：等于页大小，且能整除
# NFS 挂载的 1 MB rsize，这样一次 pread 不会被拆得不整齐。
ALIGN = 4096
O_DIRECT = getattr(os, "O_DIRECT", 0)


@dataclass
class Partition:
    """某个 safetensors 文件内的一段连续字节区间。"""

    path: Path
    offset: int
    length: int


@dataclass
class StreamResult:
    """单条流在一次运行中的观测结果。"""

    nbytes: int = 0
    # 单次 pread 的时延，单位微秒，每完成一次读追加一条。保留原始值是为了让
    # 调用方能报分位数，而不只是一个均值。
    latencies_us: list[float] = field(default_factory=list)
    # --pin-dest 时每次拷贝到 pinned 的耗时，同样保留原始值
    copy_us: list[float] = field(default_factory=list)
    error: str | None = None


# --------------------------------------------------------------------------
# 切分：把模型文件切成对齐的、按流均分的字节区间
# --------------------------------------------------------------------------


def align_down(value: int, alignment: int = ALIGN) -> int:
    return value // alignment * alignment


def build_partitions(files: Sequence[Path], nstreams: int, limit_bytes: int | None) -> list[Partition]:
    """把拼接后的模型切成 nstreams 段等长、对齐的字节区间。

    区间按文件来表示，因为 pread 需要一个确切的文件内偏移。跨越两个文件边界的
    区间会分别向两边各贡献一段，所以一条流可能同时拥有来自多个文件的片段。
    """
    sizes = [(path, path.stat().st_size) for path in files]
    total = sum(size for _, size in sizes)
    if limit_bytes:
        total = min(total, int(limit_bytes))
    total = align_down(total)

    # 先算出全局等分边界，再投影到每个文件上
    boundaries = [align_down(total * i // nstreams) for i in range(nstreams + 1)]
    boundaries[nstreams] = total

    partitions: list[Partition] = []
    consumed = 0
    for path, size in sizes:
        file_start, file_end = consumed, consumed + size
        consumed = file_end
        for i in range(nstreams):
            lo = max(boundaries[i], file_start)
            hi = align_down(min(boundaries[i + 1], file_end))
            if hi - lo >= ALIGN:
                partitions.append(Partition(path=path, offset=lo - file_start, length=hi - lo))
    return partitions


def assign_balanced(partitions: Sequence[Partition], nstreams: int) -> list[list[Partition]]:
    """用 LPT（最长优先）贪心把片段重新摊到各条流上。

    等分本身已经比较均匀，但文件边界会让片段大小参差不齐；重新摊一次可以避免
    某条流特别重、把整轮墙钟时间拖住。
    """
    buckets: list[list[Partition]] = [[] for _ in range(nstreams)]
    load = [0] * nstreams
    for part in sorted(partitions, key=lambda p: p.length, reverse=True):
        target = load.index(min(load))
        buckets[target].append(part)
        load[target] += part.length
    return buckets


def make_buffer(size: int) -> mmap.mmap:
    """开一块匿名 mmap 作为读取落地缓冲区。

    mmap.mmap(-1, n) 的起始地址按页对齐（O_DIRECT 要的就是这个），而且是普通
    匿名页，不会走进 NPU 驱动的锁页内存路径。
    """
    return mmap.mmap(-1, align_down(size))


# --------------------------------------------------------------------------
# 读循环：线程模式和进程模式共用
# --------------------------------------------------------------------------


def read_work(
    work: Sequence[Partition],
    buffer: mmap.mmap,
    chunk_bytes: int,
    repeats: int,
    use_direct: bool,
    barrier,
    result: StreamResult,
    pin_dest: bool,
) -> None:
    """把 work 里的所有片段重复 repeats 遍读进 buffer。

    buffer 故意开得很小并复用：我们要量的是 NFS 链路，不是内存占用。而且用了
    O_DIRECT 之后，复用缓冲区每一遍依然要完整走一趟服务端，不会打折。
    """
    try:
        view = memoryview(buffer)
        capacity = len(view)
        chunk_bytes = align_down(min(chunk_bytes, capacity))
        if chunk_bytes <= 0:
            raise ValueError("chunk-mb 相对缓冲区太小")
        flags = os.O_RDONLY | (O_DIRECT if use_direct else 0)

        # --pin-dest 时额外准备一块 pinned 落地区，用来量"再拷一次"的代价。
        pinned = None
        if pin_dest:
            import torch

            pinned = torch.empty(chunk_bytes, dtype=torch.uint8, pin_memory=True).numpy()

        # 所有流在同一时刻开始读，否则第一条流会抢跑，聚合带宽就没有意义了。
        barrier.wait(timeout=120)

        # 每个文件只 open 一次，并在 repeats 之间保持 fd 不关；反复 open 会带来
        # 额外的 NFS 元数据流量，那不属于我们要量的东西。
        opened: dict[Path, int] = {}
        try:
            for part in work:
                if part.path not in opened:
                    opened[part.path] = os.open(part.path, flags)
            for _ in range(repeats):
                for part in work:
                    fd = opened[part.path]
                    pos = 0
                    while pos < part.length:
                        nbytes = align_down(min(chunk_bytes, part.length - pos))
                        if nbytes <= 0:
                            break
                        start = time.perf_counter()
                        got = os.preadv(fd, [view[:nbytes]], part.offset + pos)
                        elapsed = time.perf_counter() - start
                        if got != nbytes:
                            raise OSError(f"{part.path.name} 短读：{got} != {nbytes}")
                        result.latencies_us.append(elapsed * 1e6)
                        result.nbytes += got
                        if pinned is not None:
                            start = time.perf_counter()
                            pinned[:nbytes] = view[:nbytes]
                            result.copy_us.append((time.perf_counter() - start) * 1e6)
                        pos += nbytes
        finally:
            for fd in opened.values():
                os.close(fd)
    except Exception as err:  # noqa: BLE001 - 通过 result 对象向上抛出
        result.error = f"{type(err).__name__}: {err}"
    finally:
        # 有流在到达 barrier 之前就挂了的话，其他流会一直卡在那里等；上面的
        # timeout 加上这里的 abort 一起兜住这种情况。
        try:
            barrier.abort()
        except Exception:  # noqa: BLE001 - 尽力而为，只是防死锁
            pass


def _process_entry(
    work: Sequence[Partition],
    buffer_size: int,
    chunk_bytes: int,
    repeats: int,
    use_direct: bool,
    barrier,
    pin_dest: bool,
    out: multiprocessing.Queue,
) -> None:
    """进程模式的 worker：自己持有一块 mmap 缓冲区，结果通过队列回传。"""
    buffer = make_buffer(buffer_size)
    result = StreamResult()
    read_work(work, buffer, chunk_bytes, repeats, use_direct, barrier, result, pin_dest)
    out.put((result.nbytes, result.latencies_us, result.copy_us, result.error))


# --------------------------------------------------------------------------
# 跑一个扫描点
# --------------------------------------------------------------------------


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * pct / 100))]


def run_once(args: argparse.Namespace, nstreams: int) -> dict:
    files = sorted(Path(args.model_path).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"{args.model_path} 下没有 safetensors 文件")

    limit = int(args.limit_gb * (1 << 30)) if args.limit_gb else None
    plan = assign_balanced(build_partitions(files, nstreams, limit), nstreams)
    plan_bytes = [sum(p.length for p in work) for work in plan]

    chunk_bytes = args.chunk_mb << 20
    if args.pool:
        # 真实冷加载形态：每条流一块和它负责区间等大的缓冲区，一次性装下整个
        # 模型。量出来的墙钟时间就是"整个模型从共享盘搬进来要多久"。
        buffer_size = max(align_down(max(plan_bytes)), chunk_bytes)
    else:
        # 带宽扫描形态：小块复用，只关心链路能跑多快，不关心装不装得下。
        buffer_size = chunk_bytes

    latencies: list[float] = []
    copy_us: list[float] = []
    total_bytes = 0

    start = time.perf_counter()
    if args.mode == "process":
        ctx = multiprocessing.get_context("fork")
        # 这里必须用 multiprocessing 的 Barrier，不能用 threading 的：fork 出来的
        # 子进程拿到的是 threading.Barrier 内部锁的副本，跨进程等它是不安全的。
        barrier = ctx.Barrier(nstreams)
        out: multiprocessing.Queue = ctx.Queue()
        procs = [
            ctx.Process(
                target=_process_entry,
                args=(plan[i], buffer_size, chunk_bytes, args.repeats, not args.no_direct, barrier, args.pin_dest, out),
                daemon=True,
            )
            for i in range(nstreams)
        ]
        for proc in procs:
            proc.start()
        for _ in procs:
            nbytes, lat, copied, error = out.get(timeout=args.timeout)
            if error:
                raise RuntimeError(error)
            total_bytes += nbytes
            latencies.extend(lat)
            copy_us.extend(copied)
        for proc in procs:
            proc.join()
    else:
        barrier = threading.Barrier(nstreams)
        buffers = [make_buffer(buffer_size) for _ in range(nstreams)]
        results = [StreamResult() for _ in range(nstreams)]
        threads = [
            threading.Thread(
                target=read_work,
                args=(
                    plan[i],
                    buffers[i],
                    chunk_bytes,
                    args.repeats,
                    not args.no_direct,
                    barrier,
                    results[i],
                    args.pin_dest,
                ),
                daemon=True,
            )
            for i in range(nstreams)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=args.timeout)
        errors = [r.error for r in results if r.error]
        if errors:
            raise RuntimeError("; ".join(errors))
        total_bytes = sum(r.nbytes for r in results)
        latencies = [us for r in results for us in r.latencies_us]
        copy_us = [us for r in results for us in r.copy_us]
        for buffer in buffers:
            buffer.close()
    wall = time.perf_counter() - start

    if total_bytes == 0:
        raise RuntimeError("一个字节都没读到")

    row = {
        "streams": nstreams,
        "mode": args.mode,
        "io": "buffered" if args.no_direct else "o_direct",
        "pool": args.pool,
        "repeats": args.repeats,
        "chunk_mb": args.chunk_mb,
        "bytes": total_bytes,
        "wall_s": round(wall, 4),
        "aggregate_gbps": round(total_bytes * 8 / wall / 1e9, 3),
        "aggregate_gbytes_s": round(total_bytes / wall / 1e9, 3),
        "chunks": len(latencies),
        "lat_mean_us": round(statistics.fmean(latencies), 1),
        "lat_p50_us": round(percentile(latencies, 50), 1),
        "lat_p99_us": round(percentile(latencies, 99), 1),
        # 各条流分到的字节数，用来判断负载是否摊平；差距大说明墙钟时间是被最重的
        # 那条流决定的，而不是被链路决定的。
        "plan_mb": [round(n / (1 << 20), 1) for n in plan_bytes],
    }
    if copy_us:
        row["copy_mean_us"] = round(statistics.fmean(copy_us), 1)
    return row


# --------------------------------------------------------------------------
# 自检：证明 O_DIRECT 真的在读这个文件
# --------------------------------------------------------------------------


def verify(args: argparse.Namespace) -> None:
    """同一段数据用两种方式各读一遍做比对，避免测试工具本身在说谎。

    O_DIRECT 配错了很容易出现"字节数看着正常、内容其实不对"的情况。这里先把
    一段真实数据用带缓冲的方式读出来，再用 O_DIRECT 读一遍，要求两者逐字节相同。
    """
    files = sorted(Path(args.model_path).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"{args.model_path} 下没有 safetensors 文件")
    path = files[0]
    nbytes = 1 << 20

    reference = bytearray(nbytes)
    with open(path, "rb") as handle:
        handle.seek(ALIGN)
        reference = handle.read(nbytes)

    buffer = make_buffer(nbytes)
    direct = os.open(path, os.O_RDONLY | O_DIRECT)
    try:
        got = os.preadv(direct, [memoryview(buffer)[:nbytes]], ALIGN)
        assert got == nbytes, f"O_DIRECT 短读：{got} != {nbytes}"
    finally:
        os.close(direct)

    if memoryview(buffer)[:nbytes] != reference:
        raise AssertionError("O_DIRECT 读到的内容与带缓冲读不一致")
    buffer.close()

    print(f"[PASS] O_DIRECT 与带缓冲读结果一致：{path.name}（{nbytes} 字节 @ 偏移 {ALIGN}）")


# --------------------------------------------------------------------------
# 命令行
# --------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="扫描 NFS 读带宽：多流拉取上限")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--streams", default="1,2,4,6,8,12,16,24,32", help="comma-separated stream counts")
    parser.add_argument("--repeats", type=int, default=2, help="passes over the assigned range per stream")
    parser.add_argument("--chunk-mb", type=int, default=4, help="per-pread size")
    parser.add_argument("--limit-gb", type=float, default=0.0, help="cap the working set; 0 reads the whole model")
    parser.add_argument("--mode", choices=["thread", "process"], default="thread")
    parser.add_argument("--pool", action="store_true", help="size buffers to hold the whole model, one pass, realistic cold load")
    parser.add_argument("--pin-dest", action="store_true", help="also copy each chunk into pinned memory and time it")
    parser.add_argument("--no-direct", action="store_true", help="let the page cache serve reads instead of O_DIRECT")
    parser.add_argument("--timeout", type=float, default=1800.0, help="per-sweep-point stall guard, seconds")
    parser.add_argument("--verify", action="store_true", help="check O_DIRECT against a buffered read, then exit")
    parser.add_argument("--json-out", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not O_DIRECT and not args.no_direct:
        raise SystemExit("当前平台没有 os.O_DIRECT；若确实想量 page cache，请加 --no-direct")
    if args.verify:
        verify(args)
        return 0

    rows = []
    for nstreams in [int(item) for item in args.streams.split(",") if item.strip()]:
        try:
            row = run_once(args, nstreams)
        except Exception as err:  # noqa: BLE001 - 失败一个点也要把剩下的扫完
            row = {"streams": nstreams, "error": f"{type(err).__name__}: {err}"}
        rows.append(row)
        if "error" in row:
            print(f"streams={nstreams:>3}  ERROR {row['error']}", flush=True)
        else:
            extra = f"  copy={row['copy_mean_us']:>8.1f}us" if "copy_mean_us" in row else ""
            print(
                f"streams={nstreams:>3}  {row['aggregate_gbps']:>7.2f} Gbps"
                f"  {row['aggregate_gbytes_s']:>6.2f} GB/s"
                f"  wall={row['wall_s']:>7.2f}s"
                f"  chunks={row['chunks']:>6}"
                f"  lat p50={row['lat_p50_us']:>8.1f}us"
                f"  p99={row['lat_p99_us']:>9.1f}us{extra}",
                flush=True,
            )

    best = max((r for r in rows if "error" not in r), key=lambda r: r["aggregate_gbps"], default=None)
    if best:
        print(f"\n峰值 {best['aggregate_gbps']:.2f} Gbps，出现在 {best['streams']} 条流", flush=True)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=2))
        print(f"已写入 {args.json_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
"""Transfer backend abstraction for static weight cache V1.

Two backends are available:

``tcp`` (default)
    Plain host-to-host TCP between the cache server and its clients. This is the
    backend to use here. Mooncake's CANN build installs AscendDirectTransport
    unconditionally in ``TransferEngine.initialize()``, and once installed that
    transport owns every ``batch_register_memory`` region, so transfers are
    routed over the Ascend link (ADXL) instead of TCP or RDMA. ADXL then refuses
    to connect two peers that sit on the same host and the same device with
    ``PARAM_INVALID (103900)``, which makes host-memory-to-host-memory transfer
    impossible. Registering with the RDMA transport is not an option either: the
    Ascend transport claims the registration first. Plain TCP sidesteps the
    driver entirely, which also avoids the devmm stalls that block for minutes
    on every device-memory touch.

``mooncake``
    The original backend, kept for hosts where the Ascend path is healthy.
"""

from __future__ import annotations

import ctypes
import logging
import socket
import struct
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .host_mem_manager import BucketAllocation

logger = logging.getLogger(__name__)

# Data-plane framing. A request is a one-byte opcode followed by the remote
# address to read from and the byte count; a reply is a status byte, the byte
# count actually sent, and then that many raw bytes.
_REQUEST = struct.Struct(">BQQ")
_REPLY_HEADER = struct.Struct(">BQ")
_OP_READ = 0
_STATUS_OK = 0

# Socket buffers are pushed well above the default so a single stream can keep
# the link busy without the sender stalling on every round trip.
_SOCK_BUF = 8 << 20


def ensure_ascend_device(device_id: int = 0) -> bool:
    """Best-effort ACL device selection before touching Mooncake.

    The CANN build of Mooncake installs AscendDirectTransport unconditionally in
    ``TransferEngine.initialize()``, and that path calls ``aclrtGetDevice()``
    before any device has been selected. On a fresh process that returns 107002
    and ``initialize()`` fails with ret=-1, even for a pure TCP transfer.

    A device is only claimed when none is selected yet: a process that already
    owns an NPU (a vLLM worker, say) must not be silently moved to another one.

    Returns True when a device call succeeded, False when ACL is unavailable.
    """
    try:
        acl = ctypes.CDLL("libascendcl.so")
    except OSError:
        return False

    current = ctypes.c_int32(-1)
    if acl.aclrtGetDevice(ctypes.byref(current)) == 0:
        return True
    if acl.aclrtSetDevice(ctypes.c_int32(device_id)) != 0:
        logger.warning("aclrtSetDevice(%s) failed; Mooncake init may fail", device_id)
        return False
    logger.debug("Selected ACL device %s for Mooncake Ascend transport", device_id)
    return True


@dataclass
class TransferBackendConfig:
    backend: str = "tcp"
    protocol: str = "tcp"
    device_name: str = ""
    metadata_server: str = "P2PHANDSHAKE"
    ascend_device_id: int = 0
    init_ascend_device: bool = True

    # The pool is pinned regardless of backend, and that is deliberate. Pinned
    # pages cannot be reclaimed, so the cached weights survive memory pressure;
    # a pageable pool can be evicted piecemeal under load, which would silently
    # turn the cache into a miss. Pinned memory is also what an RDMA transport
    # needs to register, so keeping the pool pinned now avoids a second copy of
    # the data flow later.


class TransferBackend(ABC):
    @abstractmethod
    def start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def peer_sid(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def register_regions(self, regions: list[tuple[int, int]]) -> None:
        """Make ``(base_ptr, capacity)`` host regions reachable by the engine.

        Both ends need this: the server registers the pool it serves from, and
        the client registers the buffer it receives into. The engine cannot write
        into a region it has not been told about.
        """
        raise NotImplementedError

    def register_buckets(self, buckets: list[BucketAllocation]) -> None:
        if not buckets:
            return
        self.register_regions([(bucket.base_ptr, bucket.capacity) for bucket in buckets])

    @abstractmethod
    def read(self, peer_sid: str, dst_ptr: int, src_ptr: int, nbytes: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class MooncakeTransferBackend(TransferBackend):
    def __init__(self, local_ip: str, config: TransferBackendConfig) -> None:
        self.local_ip = local_ip
        self.config = config
        self._engine = None

    def start(self) -> None:
        from mooncake.engine import TransferEngine

        if self.config.init_ascend_device:
            ensure_ascend_device(self.config.ascend_device_id)
        self._engine = TransferEngine()
        ret = self._engine.initialize(
            self.local_ip,
            self.config.metadata_server,
            self.config.protocol,
            self.config.device_name,
        )
        if ret != 0:
            raise RuntimeError(
                f"Failed to initialize Mooncake TransferEngine: "
                f"protocol={self.config.protocol}, ret={ret}"
            )

    def peer_sid(self) -> str:
        self._require_started()
        return f"{self.local_ip}:{self._engine.get_rpc_port()}"

    def register_regions(self, regions: list[tuple[int, int]]) -> None:
        self._require_started()
        if not regions:
            return
        bases = [int(ptr) for ptr, _ in regions]
        capacities = [int(capacity) for _, capacity in regions]
        ret = self._engine.batch_register_memory(bases, capacities)
        if ret != 0:
            raise RuntimeError(
                f"Mooncake batch_register_memory failed: ret={ret} for {len(regions)} region(s) "
                f"starting at {hex(bases[0])}. The Ascend transport only accepts Ascend-pinned "
                f"buffers; pageable or plain anonymous memory is rejected as location:*."
            )

    def read(self, peer_sid: str, dst_ptr: int, src_ptr: int, nbytes: int) -> None:
        self._require_started()
        ret = self._engine.transfer_sync_read(peer_sid, dst_ptr, src_ptr, nbytes)
        if ret != 0:
            raise RuntimeError(f"Mooncake transfer_sync_read failed: ret={ret}")

    def close(self) -> None:
        self._engine = None

    def _require_started(self) -> None:
        if self._engine is None:
            raise RuntimeError("Transfer backend has not been started")


class TcpTransferBackend(TransferBackend):
    """Serve registered host regions over TCP and read a peer's regions back.

    The server process owns the cache pool, so a client cannot address it
    directly. It asks for a byte range by the same address the metadata service
    advertised, and this side checks that range against the regions it was told
    to serve before touching memory. Nothing outside a registered region is
    reachable, which matters because the request carries a raw address.
    """

    def __init__(self, local_ip: str, config: TransferBackendConfig) -> None:
        self.local_ip = local_ip
        self.config = config
        self._regions: list[tuple[int, int]] = []
        self._listener: socket.socket | None = None
        self._port = 0
        self._accept_thread: threading.Thread | None = None
        self._handlers: set[threading.Thread] = set()
        self._stop = threading.Event()
        # One connection per peer, reused across reads; a fresh handshake for
        # every bucket would dominate the transfer time.
        self._peers: dict[str, tuple[socket.socket, threading.Lock]] = {}
        self._peers_lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((self.local_ip, 0))
        listener.listen(64)
        self._port = listener.getsockname()[1]
        self._listener = listener
        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()
        logger.info("TCP transfer backend listening on %s", self.peer_sid())

    def peer_sid(self) -> str:
        return f"{self.local_ip}:{self._port}"

    def close(self) -> None:
        self._stop.set()
        if self._listener is not None:
            # Unblocks accept() so the loop can notice the stop flag.
            try:
                self._listener.close()
            except OSError:
                pass
            self._listener = None
        with self._peers_lock:
            peers, self._peers = self._peers, {}
        for conn, _ in peers.values():
            try:
                conn.close()
            except OSError:
                pass

    # -- serving -----------------------------------------------------------

    def _accept_loop(self) -> None:
        assert self._listener is not None
        while not self._stop.is_set():
            try:
                conn, addr = self._listener.accept()
            except OSError:
                if not self._stop.is_set():
                    logger.exception("TCP transfer accept failed")
                break
            handler = threading.Thread(target=self._serve_conn, args=(conn, addr), daemon=True)
            self._handlers.add(handler)
            handler.start()

    def _serve_conn(self, conn: socket.socket, addr) -> None:
        try:
            self._tune(conn)
            while not self._stop.is_set():
                header = _recv_exact(conn, _REQUEST.size)
                if header is None:
                    break
                op, src_ptr, nbytes = _REQUEST.unpack(header)
                if op != _OP_READ or not self._is_served(src_ptr, nbytes):
                    logger.warning(
                        "Rejected transfer request from %s: op=%s ptr=%#x len=%s",
                        addr,
                        op,
                        src_ptr,
                        nbytes,
                    )
                    conn.sendall(_REPLY_HEADER.pack(1, 0))
                    break
                source = (ctypes.c_uint8 * nbytes).from_address(src_ptr)
                conn.sendall(_REPLY_HEADER.pack(_STATUS_OK, nbytes))
                conn.sendall(memoryview(source))
        except OSError as err:
            logger.debug("Transfer connection from %s ended: %s", addr, err)
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def _is_served(self, src_ptr: int, nbytes: int) -> bool:
        """Only registered ranges may be read, and only in full."""
        end = src_ptr + nbytes
        return any(src_ptr >= base and end <= base + capacity for base, capacity in self._regions)

    # -- client side -------------------------------------------------------

    def register_regions(self, regions: list[tuple[int, int]]) -> None:
        for ptr, capacity in regions:
            self._regions.append((int(ptr), int(capacity)))
        logger.debug("TCP backend serving %s region(s)", len(self._regions))

    def read(self, peer_sid: str, dst_ptr: int, src_ptr: int, nbytes: int) -> None:
        conn, lock = self._peer(peer_sid)
        with lock:
            conn.sendall(_REQUEST.pack(_OP_READ, src_ptr, nbytes))
            header = _recv_exact(conn, _REPLY_HEADER.size)
            if header is None:
                raise ConnectionError(f"peer {peer_sid} closed during transfer")
            status, actual = _REPLY_HEADER.unpack(header)
            if status != _STATUS_OK:
                raise RuntimeError(f"peer {peer_sid} rejected read of {nbytes} bytes at {hex(src_ptr)}")
            target = (ctypes.c_uint8 * actual).from_address(dst_ptr)
            _recv_into_exact(conn, target, actual)

    def _peer(self, peer_sid: str) -> tuple[socket.socket, threading.Lock]:
        with self._peers_lock:
            entry = self._peers.get(peer_sid)
            if entry is not None:
                return entry
            host, _, port = peer_sid.rpartition(":")
            conn = socket.create_connection((host, int(port)), timeout=30)
            conn.settimeout(None)
            self._tune(conn)
            entry = (conn, threading.Lock())
            self._peers[peer_sid] = entry
            return entry

    @staticmethod
    def _tune(conn: socket.socket) -> None:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, _SOCK_BUF)
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, _SOCK_BUF)


def _recv_exact(conn: socket.socket, nbytes: int) -> bytes | None:
    """Read exactly ``nbytes``; None if the peer closed cleanly at a boundary."""
    chunks: list[bytes] = []
    remaining = nbytes
    while remaining > 0:
        chunk = conn.recv(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv_into_exact(conn: socket.socket, target, nbytes: int) -> None:
    """Fill ``target`` directly, so bulk data never lands in an intermediate buffer."""
    view = memoryview(target)
    offset = 0
    while offset < nbytes:
        got = conn.recv_into(view[offset:], nbytes - offset)
        if got == 0:
            raise ConnectionError("peer closed mid-transfer")
        offset += got


def build_transfer_backend(local_ip: str, config: TransferBackendConfig) -> TransferBackend:
    if config.backend == "tcp":
        return TcpTransferBackend(local_ip=local_ip, config=config)
    if config.backend == "mooncake":
        return MooncakeTransferBackend(local_ip=local_ip, config=config)
    raise ValueError(f"Unsupported transfer backend: {config.backend}")

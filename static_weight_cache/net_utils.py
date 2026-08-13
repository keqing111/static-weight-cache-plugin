"""Small stdlib-only network helpers for the standalone cache plugin."""

from __future__ import annotations

import socket


def get_local_ip() -> str:
    """Best-effort local IP discovery without ray/verl dependencies."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        try:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def get_free_port(bind_ip: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((bind_ip, 0))
        return int(sock.getsockname()[1])

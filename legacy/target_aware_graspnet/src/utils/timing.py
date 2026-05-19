from __future__ import annotations

import time
from contextlib import contextmanager


@contextmanager
def timed(name: str, sink: dict):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        sink[name] = time.perf_counter() - t0


def now() -> float:
    return time.perf_counter()

# Copyright (c) 2025 BAAI. All rights reserved.

"""
Functional test fixtures and configuration.

Functional tests validate operator/component correctness (ops, compilation,
distributed). They require GPU but not large model files.

Note: Common fixtures (device, has_accelerator, markers) are inherited
from the root tests/conftest.py. Only functional-specific fixtures belong here.
"""


def pytest_sessionfinish(session, exitstatus):
    """Prevent inductor _read_thread segfault on Ascend NPU during interpreter teardown.

    Root cause: ATB (Ascend Tensor Backend) internally triggers torch._inductor,
    which spawns a SubprocPool with a daemon _read_thread blocked on recv().
    When Python GC later destroys the pool, the subprocess pipe closes and
    Ascend's Python 3.11 build segfaults in recv() instead of raising EOFError.

    Fix: explicitly close the subprocess stdin and join the read thread HERE,
    before Python GC runs — no performance penalty, subprocess pool still works
    normally during the test session.
    """
    import contextlib
    import threading

    try:
        from torch._inductor.compile_worker import subproc_pool as _sp
    except Exception:
        return

    # Locate the pool instance (attribute name varies across PyTorch versions)
    pool = None
    for _name in ("_pool", "_global_pool", "pool", "_worker_pool"):
        pool = getattr(_sp, _name, None)
        if pool is not None:
            break
    if pool is None:
        return

    # Step 1: close subprocess stdin → subprocess sees EOF and exits cleanly
    proc = getattr(pool, "_proc", None)
    if proc is not None:
        with contextlib.suppress(OSError):
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
        try:
            proc.wait(timeout=3.0)
        except Exception:
            with contextlib.suppress(Exception):
                proc.kill()

    # Step 2: join the read thread so it exits before GC reclaims the pipe fd
    read_thread = getattr(pool, "_read_thread", None)
    if isinstance(read_thread, threading.Thread) and read_thread.is_alive():
        read_thread.join(timeout=3.0)

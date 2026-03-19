# Copyright (c) 2025 BAAI. All rights reserved.

"""
Shared vLLM server lifecycle helper for serving E2E tests.

Provides ``VllmServer`` — a context manager that starts a vLLM serve
process, waits for readiness, and tears it down on exit.

Usage in test fixtures::

    @pytest.fixture(scope="module")
    def vllm_server():
        with VllmServer(model=MODEL_PATH, tp_size=8) as srv:
            yield srv


    def test_completion(vllm_server):
        resp = requests.post(f"{vllm_server.base_url}/completions", ...)
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass, field

import pytest
import requests


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@dataclass
class VllmServer:
    """Manages a vLLM serving process lifecycle."""

    model: str
    tp_size: int = 1
    extra_args: list[str] = field(default_factory=list)
    host: str = "127.0.0.1"
    api_key: str = ""
    max_retries: int = 60
    poll_interval: int = 5

    # Set after start
    port: int = 0
    base_url: str = ""
    _process: subprocess.Popen | None = None
    _log_file: tempfile._TemporaryFileWrapper | None = None

    def start(self) -> None:
        self.port = _get_free_port()
        self.base_url = f"http://{self.host}:{self.port}/v1"

        cmd = [
            "vllm",
            "serve",
            self.model,
            "--tensor-parallel-size",
            str(self.tp_size),
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]
        if self.api_key:
            cmd.extend(["--api-key", self.api_key])
        cmd.extend(self.extra_args)

        model_short = os.path.basename(self.model)
        print(f"\n[Setup] Starting vLLM ({model_short}, TP={self.tp_size})")
        print(f"[Setup] Command: {' '.join(cmd)}")

        self._log_file = tempfile.NamedTemporaryFile(  # noqa: SIM115
            prefix=f"vllm_{model_short}_",
            suffix=".log",
            delete=False,
        )
        self._process = subprocess.Popen(
            cmd,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        self._wait_ready()

    def stop(self) -> None:
        if self._process is None:
            return
        print("\n[Teardown] Shutting down vLLM service...")
        try:
            os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
            self._process.wait(timeout=30)
        except Exception:
            self._process.kill()
        finally:
            if self._log_file:
                self._log_file.close()
                os.unlink(self._log_file.name)
            self._process = None

    def __enter__(self) -> VllmServer:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def _wait_ready(self) -> None:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        print("[Setup] Waiting for service to be ready...")
        for i in range(self.max_retries):
            if self._process.poll() is not None:
                self._fail_with_logs(
                    "vLLM process exited unexpectedly "
                    f"(code={self._process.returncode})"
                )

            try:
                resp = requests.get(
                    f"{self.base_url}/models",
                    headers=headers,
                    timeout=5,
                )
                if resp.status_code == 200:
                    print(f"[Setup] vLLM service ready (port={self.port})")
                    return
            except requests.exceptions.ConnectionError:
                pass

            print(f"[Setup] Waiting ({i + 1}/{self.max_retries})...")
            time.sleep(self.poll_interval)

        self._fail_with_logs("vLLM service startup timed out")

    def _fail_with_logs(self, message: str) -> None:
        logs = ""
        if self._log_file:
            self._log_file.flush()
            with open(self._log_file.name) as f:
                logs = f.read()
        self.stop()
        pytest.fail(f"{message}.\nLogs (last 8000 chars):\n{logs[-8000:]}")

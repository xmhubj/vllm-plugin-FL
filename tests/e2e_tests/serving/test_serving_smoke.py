# Copyright (c) 2025 BAAI. All rights reserved.

"""
Unified serving smoke test — auto-generated from model YAML configs.

Driven by environment variables set by ``tests/run.py``:

- ``FL_TEST_MODEL``: Model family (e.g. ``qwen3``, ``minicpm``)
- ``FL_TEST_CASE``:  Case name within the family (e.g. ``next_tp8``, ``o45_tp2``)

Loads ``tests/models/<model>/<case>.yaml``, starts a vLLM server with the
engine config, and validates configured endpoints (completion and/or chat).
"""

import os

import pytest
import requests

from tests.e2e_tests.serving.server_helper import VllmServer
from tests.utils.model_config import ModelConfig

# ---------------------------------------------------------------------------
# Load config from environment (injected by run.py)
# ---------------------------------------------------------------------------

_MODEL = os.environ.get("FL_TEST_MODEL", "")
_CASE = os.environ.get("FL_TEST_CASE", "")

if not _MODEL or not _CASE:
    pytest.skip(
        "FL_TEST_MODEL and FL_TEST_CASE must be set (injected by run.py)",
        allow_module_level=True,
    )

_CFG = ModelConfig.load(_MODEL, _CASE)

pytestmark = pytest.mark.skipif(
    not os.path.exists(_CFG.model),
    reason=f"Model not found: {_CFG.model}",
)


# ---------------------------------------------------------------------------
# Server fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def server():
    """Start vLLM server with model config and optional serve overrides."""
    serve = _CFG.serve

    # Build extra_args from engine config + serve-specific overrides
    extra_args = _CFG.serve_args(**serve.extra_engine)

    with VllmServer(
        model=_CFG.model,
        tp_size=_CFG.engine.get("tensor_parallel_size", 1),
        api_key=serve.api_key,
        extra_args=extra_args,
    ) as srv:
        yield srv


@pytest.fixture
def base_url(server):
    return server.base_url


@pytest.fixture
def headers():
    serve = _CFG.serve
    h: dict[str, str] = {"Content-Type": "application/json"}
    if serve.api_key:
        h["Authorization"] = f"Bearer {serve.api_key}"
    return h


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.e2e
def test_model_list(base_url, headers):
    """Service must expose the loaded model in /v1/models."""
    response = requests.get(f"{base_url}/models", headers=headers)
    assert response.status_code == 200
    models = response.json()["data"]
    assert any(m["id"] == _CFG.model for m in models)


def _run_completion(base_url: str, headers: dict) -> None:
    """Validate /v1/completions endpoint."""
    serve = _CFG.serve
    payload = {
        "model": _CFG.model,
        "prompt": serve.completion_prompt,
        "max_tokens": serve.max_tokens,
        "temperature": 0,
    }
    response = requests.post(
        f"{base_url}/completions",
        headers=headers,
        json=payload,
    )

    assert response.status_code == 200
    data = response.json()
    assert "choices" in data
    assert len(data["choices"]) > 0

    generated_text = data["choices"][0]["text"]
    assert len(generated_text.strip()) > 0, "Generated text is empty"
    print(f"\nGenerated text: {generated_text}")


def _run_chat(base_url: str, headers: dict) -> None:
    """Validate /v1/chat/completions endpoint."""
    serve = _CFG.serve
    messages = serve.chat_messages or [
        {"role": "user", "content": "Hello"},
    ]
    payload = {
        "model": _CFG.model,
        "messages": messages,
        "max_tokens": serve.max_tokens,
    }
    response = requests.post(
        f"{base_url}/chat/completions",
        headers=headers,
        json=payload,
    )
    assert response.status_code == 200

    data = response.json()
    assert "choices" in data
    assert len(data["choices"]) > 0

    content = data["choices"][0]["message"]["content"]
    assert len(content.strip()) > 0, "Assistant message is empty"
    print(f"\nResponse: {content}")


_ENDPOINT_RUNNERS = {
    "completion": _run_completion,
    "chat": _run_chat,
}


@pytest.mark.e2e
@pytest.mark.parametrize("endpoint", _CFG.serve.endpoints, ids=_CFG.serve.endpoints)
def test_endpoint(endpoint: str, base_url, headers):
    """Validate a serving endpoint configured in the model YAML."""
    runner = _ENDPOINT_RUNNERS.get(endpoint)
    assert runner is not None, f"Unknown endpoint type: {endpoint}"
    runner(base_url, headers)

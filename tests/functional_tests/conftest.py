# Copyright (c) 2025 BAAI. All rights reserved.

"""
Functional test fixtures and configuration.
"""

import os
import pytest
import torch


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "gpu: marks tests as requiring GPU")
    config.addinivalue_line("markers", "multi_gpu: marks tests as requiring multiple GPUs")
    config.addinivalue_line("markers", "flaggems: marks tests as requiring flag_gems library")


@pytest.fixture(scope="session")
def has_gpu():
    """Check if GPU is available."""
    return torch.cuda.is_available()


@pytest.fixture(scope="session")
def device(has_gpu):
    """Get the test device."""
    if has_gpu:
        return torch.device("cuda:0")
    return torch.device("cpu")


@pytest.fixture(scope="session")
def gpu_count():
    """Get the number of available GPUs."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


@pytest.fixture
def reset_dispatch_manager():
    """Reset dispatch manager before and after test."""
    from vllm_fl.dispatch import reset_default_manager, reset_global_policy

    reset_default_manager()
    reset_global_policy()
    yield
    reset_default_manager()
    reset_global_policy()


@pytest.fixture
def clean_env():
    """Clean dispatch-related environment variables."""
    env_vars = [
        "VLLM_FL_PREFER",
        "VLLM_FL_STRICT",
        "VLLM_FL_CONFIG",
        "VLLM_FL_DENY_VENDORS",
        "VLLM_FL_ALLOW_VENDORS",
        "VLLM_FL_PER_OP",
        "VLLM_FL_DISPATCH_DEBUG",
    ]

    # Save original values
    original = {k: os.environ.get(k) for k in env_vars}

    # Clear env vars
    for k in env_vars:
        os.environ.pop(k, None)

    yield

    # Restore original values
    for k, v in original.items():
        if v is not None:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)


def skip_if_no_gpu(fn):
    """Decorator to skip test if no GPU is available."""
    return pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="GPU not available"
    )(fn)


def skip_if_no_multi_gpu(fn):
    """Decorator to skip test if less than 2 GPUs available."""
    return pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        reason="Multiple GPUs not available"
    )(fn)

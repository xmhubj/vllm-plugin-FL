# Copyright (c) 2025 BAAI. All rights reserved.

"""
Unit test fixtures and configuration.
"""

import pytest
import torch


@pytest.fixture
def mock_tensor():
    """Create a simple tensor for testing."""
    return torch.randn(2, 4, 8)


@pytest.fixture
def device():
    """Get the available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

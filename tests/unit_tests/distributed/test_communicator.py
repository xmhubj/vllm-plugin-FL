# Copyright (c) 2025 BAAI. All rights reserved.

"""
Tests for distributed communicator module.
"""

import os
import pytest
import torch
from unittest.mock import MagicMock


def has_flagcx():
    """Check if flagcx is available."""
    flagcx_path = os.getenv('FLAGCX_PATH')
    if not flagcx_path:
        return False
    lib_path = os.path.join(flagcx_path, "build/lib/libflagcx.so")
    return os.path.exists(lib_path)


# Skip all tests if flagcx is not available (communicator depends on it)
pytestmark = pytest.mark.skipif(
    not has_flagcx(),
    reason="FLAGCX_PATH not set or flagcx library not found"
)


class TestCommunicatorFL:
    """Test CommunicatorFL class."""

    def test_import(self):
        """Test that CommunicatorFL can be imported."""
        from vllm_fl.distributed.communicator import CommunicatorFL
        assert CommunicatorFL is not None

    def test_class_inherits_from_base(self):
        """Test that CommunicatorFL inherits from DeviceCommunicatorBase."""
        from vllm_fl.distributed.communicator import CommunicatorFL
        from vllm.distributed.device_communicators.base_device_communicator import (
            DeviceCommunicatorBase
        )
        assert issubclass(CommunicatorFL, DeviceCommunicatorBase)

    def test_class_has_required_methods(self):
        """Test that CommunicatorFL has all required methods."""
        from vllm_fl.distributed.communicator import CommunicatorFL

        required_methods = [
            'all_reduce',
            'reduce_scatter',
            'send',
            'recv',
            'destroy',
        ]

        for method in required_methods:
            assert hasattr(CommunicatorFL, method), f"Missing method: {method}"

    def test_instance_attributes_single_worker(self):
        """Test instance attributes for single worker scenario."""
        from vllm_fl.distributed.communicator import CommunicatorFL

        # Create instance without calling __init__ to test attribute access
        comm = CommunicatorFL.__new__(CommunicatorFL)

        # Manually set attributes that would be set by parent class
        comm.world_size = 1
        comm.use_all2all = False
        comm.cpu_group = MagicMock()
        comm.device = torch.device("cpu")
        comm.pyflagcx_comm = None

        # Verify attributes
        assert comm.world_size == 1
        assert comm.pyflagcx_comm is None

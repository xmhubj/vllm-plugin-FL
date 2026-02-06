# Copyright (c) 2025 BAAI. All rights reserved.

"""
Tests for flagcx communicator module.

Note: Most tests require FLAGCX_PATH environment variable to be set.
Tests are skipped if flagcx is not available.
"""

import os
import pytest
import torch


def has_flagcx():
    """Check if flagcx is available."""
    flagcx_path = os.getenv('FLAGCX_PATH')
    if not flagcx_path:
        return False
    lib_path = os.path.join(flagcx_path, "build/lib/libflagcx.so")
    return os.path.exists(lib_path)


# Mark all tests in this module as requiring flagcx
pytestmark = pytest.mark.skipif(
    not has_flagcx(),
    reason="FLAGCX_PATH not set or flagcx library not found"
)


class TestPyFlagcxCommunicator:
    """Test PyFlagcxCommunicator class."""

    def test_import(self):
        """Test that the module can be imported when flagcx is available."""
        from vllm_fl.distributed.device_communicators.flagcx import PyFlagcxCommunicator
        assert PyFlagcxCommunicator is not None

    def test_class_has_required_methods(self):
        """Test that PyFlagcxCommunicator has all required methods."""
        from vllm_fl.distributed.device_communicators.flagcx import PyFlagcxCommunicator

        required_methods = [
            'all_reduce',
            'all_gather',
            'reduce_scatter',
            'send',
            'recv',
            'broadcast',
            'group_start',
            'group_end',
        ]

        for method in required_methods:
            assert hasattr(PyFlagcxCommunicator, method), f"Missing method: {method}"


class TestFlagcxDataTypes:
    """Test flagcx data type related functionality."""

    def test_flagcx_dtype_enum_import(self):
        """Test that flagcxDataTypeEnum can be imported."""
        from plugin.interservice.flagcx_wrapper import flagcxDataTypeEnum
        assert flagcxDataTypeEnum is not None

    def test_flagcx_dtype_from_torch(self):
        """Test torch dtype to flagcx dtype conversion."""
        from plugin.interservice.flagcx_wrapper import flagcxDataTypeEnum

        # Test common dtypes
        test_dtypes = [
            torch.float32,
            torch.float16,
            torch.bfloat16,
        ]

        for dtype in test_dtypes:
            # Should not raise
            result = flagcxDataTypeEnum.from_torch(dtype)
            assert result is not None


class TestFlagcxReduceOps:
    """Test flagcx reduce operation types."""

    def test_flagcx_reduce_op_enum_import(self):
        """Test that flagcxRedOpTypeEnum can be imported."""
        from plugin.interservice.flagcx_wrapper import flagcxRedOpTypeEnum
        assert flagcxRedOpTypeEnum is not None

    def test_flagcx_reduce_op_from_torch(self):
        """Test torch ReduceOp to flagcx reduce op conversion."""
        from plugin.interservice.flagcx_wrapper import flagcxRedOpTypeEnum
        from torch.distributed import ReduceOp

        # Test SUM operation
        result = flagcxRedOpTypeEnum.from_torch(ReduceOp.SUM)
        assert result is not None

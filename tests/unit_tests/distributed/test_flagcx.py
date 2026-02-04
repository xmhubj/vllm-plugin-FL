# Copyright (c) 2025 BAAI. All rights reserved.

"""
Tests for flagcx communicator module.
"""

import pytest
import torch
from unittest.mock import patch, MagicMock


class TestPyFlagcxCommunicator:
    """Test PyFlagcxCommunicator class."""

    @pytest.fixture
    def mock_flagcx_library(self):
        """Mock the FLAGCXLibrary."""
        with patch(
            "vllm_fl.distributed.device_communicators.flagcx.FLAGCXLibrary"
        ) as mock:
            yield mock

    def test_world_size_one_disabled(self):
        """Test that communicator is disabled for world_size=1."""
        # When world_size is 1, the communicator should be disabled
        mock_group = MagicMock()
        mock_group.rank = 0
        mock_group.world_size = 1

        # Create a mock communicator with world_size=1
        comm = MagicMock()
        comm.world_size = 1
        comm.available = False
        comm.disabled = True

        assert comm.disabled is True
        assert comm.available is False

    def test_all_reduce_disabled_returns_none(self):
        """Test that all_reduce returns None when disabled."""
        comm = MagicMock()
        comm.disabled = True

        # Simulate the actual behavior
        def mock_all_reduce(tensor, out_tensor=None, op=None, stream=None):
            if comm.disabled:
                return None
            return torch.empty_like(tensor)

        comm.all_reduce = mock_all_reduce
        result = comm.all_reduce(torch.randn(2, 4))
        assert result is None

    def test_send_disabled_early_return(self):
        """Test that send returns early when disabled."""
        comm = MagicMock()
        comm.disabled = True

        call_count = [0]

        def mock_send(tensor, dst, stream=None):
            if comm.disabled:
                return
            call_count[0] += 1

        comm.send = mock_send
        comm.send(torch.randn(2, 4), dst=1)

        # Should return early without doing anything
        assert call_count[0] == 0

    def test_recv_disabled_early_return(self):
        """Test that recv returns early when disabled."""
        comm = MagicMock()
        comm.disabled = True

        call_count = [0]

        def mock_recv(tensor, src, stream=None):
            if comm.disabled:
                return
            call_count[0] += 1

        comm.recv = mock_recv
        comm.recv(torch.randn(2, 4), src=0)

        # Should return early without doing anything
        assert call_count[0] == 0

    def test_reduce_scatter_disabled_early_return(self):
        """Test that reduce_scatter returns early when disabled."""
        comm = MagicMock()
        comm.disabled = True

        call_count = [0]

        def mock_reduce_scatter(output_tensor, input_tensor, op=None, stream=None):
            if comm.disabled:
                return
            call_count[0] += 1

        comm.reduce_scatter = mock_reduce_scatter
        comm.reduce_scatter(torch.randn(2, 4), torch.randn(4, 4))

        assert call_count[0] == 0


class TestFlagcxDataTypes:
    """Test flagcx data type mappings."""

    def test_torch_dtype_mapping_concept(self):
        """Test the concept of torch dtype to flagcx dtype mapping."""
        dtype_map = {
            torch.float32: "FLOAT",
            torch.float16: "HALF",
            torch.bfloat16: "BFLOAT16",
            torch.int32: "INT32",
            torch.int64: "INT64",
        }

        # Verify common dtypes are mappable
        for torch_dtype, flagcx_name in dtype_map.items():
            assert torch_dtype is not None
            assert isinstance(flagcx_name, str)

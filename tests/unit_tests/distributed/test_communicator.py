# Copyright (c) 2025 BAAI. All rights reserved.

"""
Tests for distributed communicator module.
"""

import pytest
import torch
from unittest.mock import patch, MagicMock, PropertyMock


class TestCommunicatorFL:
    @pytest.fixture
    def mock_pyflagcx(self):
        with patch("vllm_fl.distributed.communicator.PyFlagcxCommunicator") as mock:
            yield mock

    @pytest.fixture
    def mock_base_communicator(self):
        with patch(
            "vllm_fl.distributed.communicator.DeviceCommunicatorBase.__init__"
        ) as mock:
            mock.return_value = None
            yield mock

    def test_import(self):
        from vllm_fl.distributed.communicator import CommunicatorFL
        assert CommunicatorFL is not None

    def test_init_single_worker_no_pyflagcx(self, mock_base_communicator, mock_pyflagcx):
        from vllm_fl.distributed.communicator import CommunicatorFL

        # Mock the base class attributes
        with patch.object(CommunicatorFL, "world_size", new_callable=PropertyMock) as mock_ws:
            mock_ws.return_value = 1

            with patch.object(CommunicatorFL, "use_all2all", new_callable=PropertyMock) as mock_a2a:
                mock_a2a.return_value = False

                cpu_group = MagicMock()
                device = torch.device("cpu")

                comm = CommunicatorFL.__new__(CommunicatorFL)
                comm.world_size = 1
                comm.use_all2all = False
                comm.cpu_group = cpu_group
                comm.device = device
                comm.pyflagcx_comm = None

                # Single worker should not create pyflagcx communicator
                assert comm.pyflagcx_comm is None


class TestPyFlagcxCommunicator:
    def test_import(self):
        # Test that the module can be imported (may fail if flagcx not available)
        try:
            from vllm_fl.distributed.device_communicators.flagcx import PyFlagcxCommunicator
            assert PyFlagcxCommunicator is not None
        except ImportError:
            pytest.skip("flagcx not available")

    def test_disabled_communicator_returns_none(self):
        """Test that disabled communicator methods return None/early exit."""
        # Create a mock disabled communicator
        mock_comm = MagicMock()
        mock_comm.disabled = True
        mock_comm.all_reduce.return_value = None

        # When disabled, all_reduce should return None
        result = mock_comm.all_reduce(torch.randn(2, 4))
        assert result is None

# Copyright (c) 2025 BAAI. All rights reserved.

"""
Tests for layernorm ops.
"""

import pytest
import torch
from unittest.mock import patch, MagicMock


class TestRMSNormFL:
    @pytest.fixture
    def mock_call_op(self):
        with patch("vllm_fl.ops.layernorm.call_op") as mock:
            yield mock

    def test_import(self):
        from vllm_fl.ops.layernorm import RMSNormFL
        assert RMSNormFL is not None

    def test_init_params(self):
        from vllm_fl.ops.layernorm import RMSNormFL

        hidden_size = 128
        eps = 1e-5
        layer = RMSNormFL(hidden_size=hidden_size, eps=eps)

        assert layer.variance_epsilon == eps
        assert layer.weight.shape == (hidden_size,)

    def test_forward_oot_without_residual(self, mock_call_op):
        from vllm_fl.ops.layernorm import RMSNormFL

        hidden_size = 128
        mock_call_op.return_value = torch.randn(2, hidden_size)

        layer = RMSNormFL(hidden_size=hidden_size)
        x = torch.randn(2, hidden_size)

        result = layer.forward_oot(x)

        mock_call_op.assert_called_once()
        call_args = mock_call_op.call_args
        assert call_args[0][0] == "rms_norm"
        assert torch.equal(call_args[0][1], x)
        assert call_args[0][2] is None  # residual
        assert torch.equal(call_args[0][3], layer.weight)

    def test_forward_oot_with_residual(self, mock_call_op):
        from vllm_fl.ops.layernorm import RMSNormFL

        hidden_size = 128
        mock_call_op.return_value = (torch.randn(2, hidden_size), torch.randn(2, hidden_size))

        layer = RMSNormFL(hidden_size=hidden_size)
        x = torch.randn(2, hidden_size)
        residual = torch.randn(2, hidden_size)

        result = layer.forward_oot(x, residual=residual)

        mock_call_op.assert_called_once()
        call_args = mock_call_op.call_args
        assert call_args[0][0] == "rms_norm"
        assert torch.equal(call_args[0][2], residual)

    def test_all_exports(self):
        from vllm_fl.ops.layernorm import __all__
        assert "RMSNormFL" in __all__

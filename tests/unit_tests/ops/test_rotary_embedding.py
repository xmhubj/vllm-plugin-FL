# Copyright (c) 2025 BAAI. All rights reserved.

"""
Tests for rotary embedding ops.
"""

import pytest
import torch
from unittest.mock import patch, MagicMock


class TestRotaryEmbeddingFL:
    @pytest.fixture
    def mock_call_op(self):
        with patch("vllm_fl.ops.rotary_embedding.call_op") as mock:
            yield mock

    def test_import(self):
        from vllm_fl.ops.rotary_embedding import RotaryEmbeddingFL
        assert RotaryEmbeddingFL is not None

    def test_init_params(self):
        from vllm_fl.ops.rotary_embedding import RotaryEmbeddingFL

        head_size = 64
        rotary_dim = 32
        max_position_embeddings = 2048
        base = 10000.0

        layer = RotaryEmbeddingFL(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_position_embeddings,
            base=base,
            is_neox_style=True,
            dtype=torch.float32,
        )

        assert layer.head_size == head_size
        assert layer.rotary_dim == rotary_dim
        assert layer.max_position_embeddings == max_position_embeddings
        assert layer.is_neox_style is True

    def test_all_exports(self):
        from vllm_fl.ops.rotary_embedding import __all__
        assert "RotaryEmbeddingFL" in __all__

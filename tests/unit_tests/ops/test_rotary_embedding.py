# Copyright (c) 2025 BAAI. All rights reserved.

"""
Tests for rotary embedding ops.
"""

import pytest
import torch
from unittest.mock import patch, MagicMock


class TestRotaryEmbeddingFL:
    """Test RotaryEmbeddingFL class."""

    @pytest.fixture
    def mock_call_op(self):
        """Mock the call_op function."""
        with patch("vllm_fl.ops.rotary_embedding.call_op") as mock:
            yield mock

    @pytest.fixture
    def mock_parent_init(self):
        """Mock the parent class __init__ to avoid vllm C++ dependencies."""
        with patch("vllm_fl.ops.rotary_embedding.RotaryEmbedding.__init__", return_value=None):
            yield

    def test_import(self):
        """Test that RotaryEmbeddingFL can be imported."""
        from vllm_fl.ops.rotary_embedding import RotaryEmbeddingFL
        assert RotaryEmbeddingFL is not None

    def test_class_exists(self):
        """Test that RotaryEmbeddingFL is a class."""
        from vllm_fl.ops.rotary_embedding import RotaryEmbeddingFL
        assert isinstance(RotaryEmbeddingFL, type)

    def test_has_forward_oot_method(self):
        """Test that RotaryEmbeddingFL has forward_oot method."""
        from vllm_fl.ops.rotary_embedding import RotaryEmbeddingFL
        assert hasattr(RotaryEmbeddingFL, 'forward_oot')
        assert callable(getattr(RotaryEmbeddingFL, 'forward_oot'))

    def test_init_calls_parent(self, mock_parent_init):
        """Test that __init__ calls parent class."""
        from vllm_fl.ops.rotary_embedding import RotaryEmbeddingFL

        layer = RotaryEmbeddingFL(
            head_size=64,
            rotary_dim=32,
            max_position_embeddings=2048,
            base=10000.0,
            is_neox_style=True,
            dtype=torch.float32,
        )

        # Instance should be created
        assert layer is not None

    def test_forward_oot_calls_dispatch(self, mock_parent_init, mock_call_op):
        """Test that forward_oot calls the dispatch call_op."""
        from vllm_fl.ops.rotary_embedding import RotaryEmbeddingFL

        # Create layer with mocked parent
        layer = RotaryEmbeddingFL(
            head_size=64,
            rotary_dim=32,
            max_position_embeddings=2048,
            base=10000.0,
            is_neox_style=True,
            dtype=torch.float32,
        )

        # Manually set attributes that parent __init__ would set
        layer.head_size = 64
        layer.rotary_dim = 32
        layer.is_neox_style = True
        layer.cos_sin_cache = torch.randn(2048, 64)

        # Setup mock return value
        mock_call_op.return_value = (torch.randn(4, 8, 32), torch.randn(4, 8, 32))

        # Call forward_oot
        positions = torch.tensor([0, 1, 2, 3])
        query = torch.randn(4, 8, 64)
        key = torch.randn(4, 8, 64)

        result = layer.forward_oot(positions, query, key)

        # Verify call_op was called with "rotary_embedding"
        mock_call_op.assert_called_once()
        call_args = mock_call_op.call_args
        assert call_args[0][0] == "rotary_embedding"

    def test_all_exports(self):
        """Test that __all__ contains RotaryEmbeddingFL."""
        from vllm_fl.ops.rotary_embedding import __all__
        assert "RotaryEmbeddingFL" in __all__

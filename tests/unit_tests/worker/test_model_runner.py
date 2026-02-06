# Copyright (c) 2025 BAAI. All rights reserved.

"""
Tests for model runner module.

Note: These tests require vllm >= 0.13.0 with full installation.
"""

import pytest
from unittest.mock import patch, MagicMock


def has_vllm_model_runner():
    """Check if vllm model runner dependencies are available."""
    try:
        from vllm_fl.worker.model_runner import ModelRunnerFL
        return True
    except ImportError:
        return False


# Skip all tests if vllm model runner is not available
pytestmark = pytest.mark.skipif(
    not has_vllm_model_runner(),
    reason="vllm_fl.worker.model_runner not available"
)


class TestModelRunnerFL:
    """Test ModelRunnerFL class."""

    def test_import(self):
        """Test that ModelRunnerFL can be imported."""
        from vllm_fl.worker.model_runner import ModelRunnerFL
        assert ModelRunnerFL is not None

    def test_is_class(self):
        """Test that ModelRunnerFL is a class."""
        from vllm_fl.worker.model_runner import ModelRunnerFL
        assert isinstance(ModelRunnerFL, type)

    def test_inherits_from_mixins(self):
        """Test that ModelRunnerFL inherits from expected mixins."""
        from vllm_fl.worker.model_runner import ModelRunnerFL
        from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin

        # ModelRunnerFL uses mixins, not GPUModelRunner
        assert issubclass(ModelRunnerFL, LoRAModelRunnerMixin)

    def test_has_load_model_method(self):
        """Test that ModelRunnerFL has load_model method."""
        from vllm_fl.worker.model_runner import ModelRunnerFL
        assert hasattr(ModelRunnerFL, 'load_model')
        assert callable(getattr(ModelRunnerFL, 'load_model'))

    def test_has_execute_model_method(self):
        """Test that ModelRunnerFL has execute_model method."""
        from vllm_fl.worker.model_runner import ModelRunnerFL
        assert hasattr(ModelRunnerFL, 'execute_model')
        assert callable(getattr(ModelRunnerFL, 'execute_model'))

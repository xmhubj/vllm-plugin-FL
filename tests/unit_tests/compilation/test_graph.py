# Copyright (c) 2025 BAAI. All rights reserved.

"""
Tests for compilation graph module.
"""

import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass


class TestGraphClasses:
    """Test graph-related classes."""

    def test_graph_entry_import(self):
        from vllm_fl.compilation.graph import GraphEntry
        assert GraphEntry is not None

    def test_graph_options_import(self):
        from vllm_fl.compilation.graph import GraphOptions
        assert GraphOptions is not None

    def test_graph_wrapper_import(self):
        from vllm_fl.compilation.graph import GraphWrapper
        assert GraphWrapper is not None


class TestGraphOptions:
    """Test GraphOptions dataclass."""

    def test_default_values(self):
        from vllm_fl.compilation.graph import GraphOptions

        options = GraphOptions()

        assert options.debug_log_enable is True
        assert options.gc_disable is False
        assert options.weak_ref_output is True

    def test_custom_values(self):
        from vllm_fl.compilation.graph import GraphOptions

        options = GraphOptions(
            debug_log_enable=False,
            gc_disable=True,
            weak_ref_output=False,
        )

        assert options.debug_log_enable is False
        assert options.gc_disable is True
        assert options.weak_ref_output is False


class TestGraphEntry:
    """Test GraphEntry dataclass."""

    def test_default_values(self):
        from vllm_fl.compilation.graph import GraphEntry

        # Create a mock BatchDescriptor
        mock_batch_desc = MagicMock()

        entry = GraphEntry(batch_descriptor=mock_batch_desc)

        assert entry.batch_descriptor is mock_batch_desc
        assert entry.graph is None
        assert entry.output is None
        assert entry.input_addresses is None


class TestWeakRefTensors:
    """Test weak_ref_tensors function."""

    def test_import(self):
        from vllm_fl.compilation.graph import weak_ref_tensors
        assert weak_ref_tensors is not None

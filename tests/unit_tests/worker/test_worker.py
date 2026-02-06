# Copyright (c) 2025 BAAI. All rights reserved.

"""
Tests for worker module.

Note: These tests require vllm >= 0.13.0 with profiler support.
"""

import pytest
from unittest.mock import patch, MagicMock


def has_vllm_profiler():
    """Check if vllm profiler is available."""
    try:
        from vllm.profiler.wrapper import TorchProfilerWrapper
        return True
    except ImportError:
        return False


# Skip all tests if vllm profiler is not available
pytestmark = pytest.mark.skipif(
    not has_vllm_profiler(),
    reason="vllm.profiler.wrapper not available (requires vllm >= 0.13.0)"
)


class TestWorkerFL:
    """Test WorkerFL class."""

    def test_import(self):
        """Test that WorkerFL can be imported."""
        from vllm_fl.worker.worker import WorkerFL
        assert WorkerFL is not None

    def test_memory_snapshot_import(self):
        """Test that MemorySnapshot can be imported."""
        from vllm_fl.worker.worker import MemorySnapshot
        assert MemorySnapshot is not None

    def test_memory_profiling_result_import(self):
        """Test that MemoryProfilingResult can be imported."""
        from vllm_fl.worker.worker import MemoryProfilingResult
        assert MemoryProfilingResult is not None


class TestMemorySnapshot:
    """Test MemorySnapshot dataclass."""

    def test_default_values(self):
        """Test that MemorySnapshot has correct default values."""
        from vllm_fl.worker.worker import MemorySnapshot

        # Create with auto_measure=False to avoid actual GPU calls
        snapshot = MemorySnapshot(auto_measure=False)

        assert snapshot.torch_peak == 0
        assert snapshot.free_memory == 0
        assert snapshot.total_memory == 0
        assert snapshot.cuda_memory == 0
        assert snapshot.torch_memory == 0
        assert snapshot.non_torch_memory == 0

    def test_subtraction(self):
        """Test MemorySnapshot subtraction operator."""
        from vllm_fl.worker.worker import MemorySnapshot

        snapshot1 = MemorySnapshot(auto_measure=False)
        snapshot1.torch_peak = 1000
        snapshot1.free_memory = 5000
        snapshot1.total_memory = 10000
        snapshot1.cuda_memory = 5000
        snapshot1.torch_memory = 3000
        snapshot1.non_torch_memory = 2000
        snapshot1.timestamp = 10.0

        snapshot2 = MemorySnapshot(auto_measure=False)
        snapshot2.torch_peak = 500
        snapshot2.free_memory = 6000
        snapshot2.total_memory = 10000
        snapshot2.cuda_memory = 4000
        snapshot2.torch_memory = 2000
        snapshot2.non_torch_memory = 2000
        snapshot2.timestamp = 5.0

        diff = snapshot1 - snapshot2

        assert diff.torch_peak == 500
        assert diff.free_memory == -1000
        assert diff.cuda_memory == 1000
        assert diff.torch_memory == 1000
        assert diff.timestamp == 5.0


class TestMemoryProfilingResult:
    """Test MemoryProfilingResult dataclass."""

    def test_default_values(self):
        """Test that MemoryProfilingResult has correct default values."""
        from vllm_fl.worker.worker import MemoryProfilingResult

        result = MemoryProfilingResult()

        assert result.weights_memory == 0
        assert result.torch_peak_increase == 0
        assert result.non_torch_increase == 0
        assert result.non_kv_cache_memory == 0
        assert result.profile_time == 0.0

    def test_before_profile_defaults(self):
        """Test that MemoryProfilingResult creates default snapshots."""
        from vllm_fl.worker.worker import MemoryProfilingResult

        result = MemoryProfilingResult()

        # Should create default MemorySnapshot objects
        assert result.before_profile is not None
        assert result.after_profile is not None


class TestInitWorkerDistributedEnvironment:
    """Test init_worker_distributed_environment function."""

    def test_import(self):
        """Test that init_worker_distributed_environment can be imported."""
        from vllm_fl.worker.worker import init_worker_distributed_environment
        assert init_worker_distributed_environment is not None

    def test_is_callable(self):
        """Test that init_worker_distributed_environment is callable."""
        from vllm_fl.worker.worker import init_worker_distributed_environment
        assert callable(init_worker_distributed_environment)

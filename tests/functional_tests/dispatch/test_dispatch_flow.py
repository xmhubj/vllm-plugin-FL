# Copyright (c) 2025 BAAI. All rights reserved.

"""
Functional tests for dispatch flow.
Tests the complete operator dispatch mechanism including
registration, resolution, and policy-based selection.
"""

import os
import pytest
import tempfile

from vllm_fl.dispatch import (
    OpRegistry,
    OpManager,
    OpImpl,
    BackendImplKind,
    BackendPriority,
    SelectionPolicy,
    get_default_manager,
    reset_default_manager,
    call_op,
    resolve_op,
    get_policy,
    set_global_policy,
    reset_global_policy,
    policy_context,
    with_preference,
    PREFER_DEFAULT,
    PREFER_VENDOR,
    PREFER_REFERENCE,
)


class TestDispatchManagerInitialization:
    """Test OpManager initialization and registration."""

    @pytest.fixture(autouse=True)
    def setup(self, reset_dispatch_manager, clean_env):
        """Reset manager before each test."""
        pass

    def test_manager_initializes_lazily(self):
        """Test that manager initializes on first use."""
        manager = get_default_manager()
        assert manager is not None

        # Should be initialized after first call
        manager.ensure_initialized()
        assert manager._state.initialized is True

    def test_manager_registers_builtin_ops(self):
        """Test that built-in operators are registered."""
        manager = get_default_manager()
        manager.ensure_initialized()

        snap = manager.registry.snapshot()

        # Should have some operators registered
        assert len(snap.impls_by_op) > 0

    def test_manager_singleton(self):
        """Test that get_default_manager returns singleton."""
        manager1 = get_default_manager()
        manager2 = get_default_manager()
        assert manager1 is manager2

    def test_reset_manager(self):
        """Test that reset_default_manager creates new instance."""
        manager1 = get_default_manager()
        reset_default_manager()
        manager2 = get_default_manager()
        assert manager1 is not manager2


class TestOperatorResolution:
    """Test operator resolution logic."""

    @pytest.fixture
    def custom_manager(self):
        """Create a custom manager with test implementations."""
        registry = OpRegistry()

        # Register test implementations
        registry.register_impl(OpImpl(
            op_name="test_op",
            impl_id="default.flagos",
            kind=BackendImplKind.DEFAULT,
            fn=lambda x: x * 2,
            priority=BackendPriority.DEFAULT,
        ))

        registry.register_impl(OpImpl(
            op_name="test_op",
            impl_id="reference.pytorch",
            kind=BackendImplKind.REFERENCE,
            fn=lambda x: x * 2 + 1,
            priority=BackendPriority.REFERENCE,
        ))

        registry.register_impl(OpImpl(
            op_name="test_op",
            impl_id="vendor.cuda",
            kind=BackendImplKind.VENDOR,
            fn=lambda x: x * 3,
            vendor="CUDA",
            priority=BackendPriority.VENDOR,
        ))

        return OpManager(registry)

    @pytest.fixture(autouse=True)
    def setup(self, reset_dispatch_manager, clean_env):
        pass

    def test_resolve_selects_by_default_order(self, custom_manager):
        """Test that resolve selects by default order (flagos first)."""
        fn = custom_manager.resolve("test_op")
        result = fn(10)
        assert result == 20  # flagos: x * 2

    def test_resolve_with_vendor_preference(self, custom_manager):
        """Test resolution with vendor preference."""
        vendor_policy = SelectionPolicy(prefer=PREFER_VENDOR)
        with policy_context(vendor_policy):
            fn = custom_manager.resolve("test_op")
            result = fn(10)
            assert result == 30  # vendor: x * 3

    def test_resolve_with_reference_preference(self, custom_manager):
        """Test resolution with reference preference."""
        ref_policy = SelectionPolicy(prefer=PREFER_REFERENCE)
        with policy_context(ref_policy):
            fn = custom_manager.resolve("test_op")
            result = fn(10)
            assert result == 21  # reference: x * 2 + 1

    def test_resolve_caches_result(self, custom_manager):
        """Test that resolution is cached."""
        fn1 = custom_manager.resolve("test_op")
        fn2 = custom_manager.resolve("test_op")
        assert fn1 is fn2

    def test_resolve_nonexistent_raises(self, custom_manager):
        """Test that resolving non-existent op raises."""
        with pytest.raises(RuntimeError, match="No available implementation"):
            custom_manager.resolve("nonexistent_op")


class TestPolicyBasedSelection:
    """Test policy-based operator selection."""

    @pytest.fixture
    def manager_with_vendors(self):
        """Create manager with multiple vendor implementations."""
        registry = OpRegistry()

        registry.register_impl(OpImpl(
            op_name="multi_vendor_op",
            impl_id="vendor.cuda",
            kind=BackendImplKind.VENDOR,
            fn=lambda: "cuda",
            vendor="CUDA",
            priority=100,
        ))

        registry.register_impl(OpImpl(
            op_name="multi_vendor_op",
            impl_id="vendor.ascend",
            kind=BackendImplKind.VENDOR,
            fn=lambda: "ascend",
            vendor="ASCEND",
            priority=100,
        ))

        registry.register_impl(OpImpl(
            op_name="multi_vendor_op",
            impl_id="reference.pytorch",
            kind=BackendImplKind.REFERENCE,
            fn=lambda: "reference",
            priority=50,
        ))

        return OpManager(registry)

    @pytest.fixture(autouse=True)
    def setup(self, reset_dispatch_manager, clean_env):
        pass

    def test_deny_vendor_excludes_implementation(self, manager_with_vendors):
        """Test that denied vendors are excluded."""
        policy = SelectionPolicy.from_dict(
            prefer=PREFER_VENDOR,
            deny_vendors={"CUDA"},
        )
        with policy_context(policy):
            fn = manager_with_vendors.resolve("multi_vendor_op")
            result = fn()
            assert result == "ascend"

    def test_allow_vendor_limits_selection(self, manager_with_vendors):
        """Test that allow list limits vendor selection."""
        policy = SelectionPolicy.from_dict(
            prefer=PREFER_VENDOR,
            allow_vendors={"CUDA"},
        )
        with policy_context(policy):
            fn = manager_with_vendors.resolve("multi_vendor_op")
            result = fn()
            assert result == "cuda"

    def test_per_op_order_overrides_default(self, manager_with_vendors):
        """Test that per-op order overrides default."""
        policy = SelectionPolicy.from_dict(
            prefer=PREFER_VENDOR,
            per_op_order={"multi_vendor_op": ["reference"]},
        )
        with policy_context(policy):
            fn = manager_with_vendors.resolve("multi_vendor_op")
            result = fn()
            assert result == "reference"


class TestFallbackMechanism:
    """Test fallback when primary implementation fails."""

    @pytest.fixture
    def manager_with_failing_impl(self):
        """Create manager with a failing primary implementation."""
        registry = OpRegistry()

        def failing_fn():
            raise RuntimeError("Primary failed!")

        registry.register_impl(OpImpl(
            op_name="fallback_op",
            impl_id="default.flagos",
            kind=BackendImplKind.DEFAULT,
            fn=failing_fn,
            priority=BackendPriority.DEFAULT,
        ))

        registry.register_impl(OpImpl(
            op_name="fallback_op",
            impl_id="reference.pytorch",
            kind=BackendImplKind.REFERENCE,
            fn=lambda: "fallback_success",
            priority=BackendPriority.REFERENCE,
        ))

        return OpManager(registry)

    @pytest.fixture(autouse=True)
    def setup(self, reset_dispatch_manager, clean_env):
        pass

    def test_fallback_on_primary_failure(self, manager_with_failing_impl):
        """Test fallback to next implementation when primary fails."""
        # Enable fallback (VLLM_FL_STRICT != "0")
        os.environ["VLLM_FL_STRICT"] = "1"

        result = manager_with_failing_impl.call("fallback_op")
        assert result == "fallback_success"

    def test_failed_impl_tracked(self, manager_with_failing_impl):
        """Test that failed implementations are tracked."""
        os.environ["VLLM_FL_STRICT"] = "1"

        manager_with_failing_impl.call("fallback_op")

        failed = manager_with_failing_impl.get_failed_impls("fallback_op")
        assert "default.flagos" in failed.get("fallback_op", set())

    def test_clear_failed_impls(self, manager_with_failing_impl):
        """Test clearing failed implementations cache."""
        os.environ["VLLM_FL_STRICT"] = "1"

        manager_with_failing_impl.call("fallback_op")
        manager_with_failing_impl.clear_failed_impls("fallback_op")

        failed = manager_with_failing_impl.get_failed_impls("fallback_op")
        assert len(failed) == 0


class TestConfigFileLoading:
    """Test loading configuration from YAML file."""

    @pytest.fixture(autouse=True)
    def setup(self, reset_dispatch_manager, clean_env):
        pass

    def test_load_config_from_yaml(self):
        """Test loading policy from YAML config file."""
        config_content = """
prefer: vendor
strict: true
allow_vendors:
  - CUDA
deny_vendors:
  - AMD
op_backends:
  test_op:
    - vendor
    - reference
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(config_content)
            config_path = f.name

        try:
            os.environ["VLLM_FL_CONFIG"] = config_path
            reset_global_policy()

            policy = get_policy()

            assert policy.prefer == "vendor"
            assert policy.strict is True
            assert "CUDA" in policy.allow_vendors
            assert "AMD" in policy.deny_vendors
            assert policy.get_per_op_order("test_op") == ["vendor", "reference"]
        finally:
            os.unlink(config_path)
            os.environ.pop("VLLM_FL_CONFIG", None)


class TestContextManagers:
    """Test policy context managers."""

    @pytest.fixture(autouse=True)
    def setup(self, reset_dispatch_manager, clean_env):
        pass

    def test_with_preference_context(self):
        """Test with_preference context manager."""
        original = get_policy()
        assert original.prefer == PREFER_DEFAULT

        with with_preference(PREFER_VENDOR):
            inside = get_policy()
            assert inside.prefer == PREFER_VENDOR

        after = get_policy()
        assert after.prefer == PREFER_DEFAULT

    def test_nested_contexts(self):
        """Test nested policy contexts."""
        with with_preference(PREFER_VENDOR):
            assert get_policy().prefer == PREFER_VENDOR

            with with_preference(PREFER_REFERENCE):
                assert get_policy().prefer == PREFER_REFERENCE

            assert get_policy().prefer == PREFER_VENDOR

        assert get_policy().prefer == PREFER_DEFAULT

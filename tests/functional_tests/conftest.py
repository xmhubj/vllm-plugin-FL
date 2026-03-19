# Copyright (c) 2025 BAAI. All rights reserved.

"""
Functional test fixtures and configuration.

Functional tests validate operator/component correctness (ops, compilation,
distributed). They require GPU but not large model files.

Note: Common fixtures (device, has_accelerator, markers) are inherited
from the root tests/conftest.py. Only functional-specific fixtures belong here.
"""

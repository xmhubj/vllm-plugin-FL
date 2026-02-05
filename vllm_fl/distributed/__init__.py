# Copyright (c) 2025 BAAI. All rights reserved.

__all__ = ["communicator"]


def __getattr__(name):
    if name == "communicator":
        from vllm_fl.distributed import communicator
        return communicator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

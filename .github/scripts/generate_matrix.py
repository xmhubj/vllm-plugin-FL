#!/usr/bin/env python3
# Copyright (c) 2025 BAAI. All rights reserved.

"""Generate test matrices from a platform config.

Reads ``tests/platforms/<platform>.yaml``, expands the per-device test
definitions into flat JSON arrays, and writes them to ``$GITHUB_OUTPUT``
so that downstream GitHub Actions jobs can use ``fromJson()`` to fan out.

Usage (in a workflow step)::

    - id: matrix
      run: python .github/scripts/generate_matrix.py --platform ${{ inputs.platform }}

Outputs:
    functional — JSON array of ``{device, timeout}`` objects (one per device, runs all functional tests).
    e2e        — JSON array of ``{task, model, case, device, timeout}`` objects (inference, serving).
    unit       — JSON array of ``{device, include, exclude}`` objects.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PLATFORMS_DIR = REPO_ROOT / "tests" / "platforms"

# YAML task key → test directory name
TASK_DIR_MAP: dict[str, str] = {
    "serve": "serving",
}

# Default timeout (minutes) per task category
DEFAULT_TIMEOUT: dict[str, int] = {
    "inference": 60,
    "serving": 60,
}


def load_platform(platform: str) -> dict:
    """Load and return the platform YAML config."""
    path = PLATFORMS_DIR / f"{platform}.yaml"
    if not path.exists():
        print(f"::error::Platform config not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_task_dir(task_key: str) -> str:
    """Map a YAML task key to the corresponding test directory name."""
    return TASK_DIR_MAP.get(task_key, task_key)


def get_device_sections(config: dict) -> list[str]:
    """Return device section names present in the config.

    Device sections are top-level keys that have a nested ``tests`` mapping.
    """
    devices = []
    for key, value in config.items():
        if isinstance(value, dict) and "tests" in value:
            devices.append(key)
    return devices


def build_e2e_matrix(
    config: dict,
    devices: list[str],
    unsupported: list[str],
) -> list[dict]:
    """End-to-end model tests (inference, serving).

    These are slow and require model files. One matrix entry per model/case.
    """
    entries = []
    for device in devices:
        section = config[device]
        e2e = section.get("tests", {}).get("e2e", {})

        for task_key, models in e2e.items():
            if not isinstance(models, dict):
                continue
            task_dir = resolve_task_dir(task_key)
            timeout = DEFAULT_TIMEOUT.get(task_dir, 60)

            for model, cases in models.items():
                # Skip models matching unsupported features
                if any(feat in model for feat in unsupported):
                    continue

                if not isinstance(cases, list):
                    cases = [cases]

                for case in cases:
                    entries.append(
                        {
                            "task": task_dir,
                            "model": model,
                            "case": str(case),
                            "device": device,
                            "timeout": timeout,
                        }
                    )
    return entries


def build_unit_matrix(config: dict, devices: list[str]) -> list[dict]:
    """Expand per-device unit test definitions into a flat matrix."""
    matrix = []
    for device in devices:
        section = config[device]
        unit = section.get("tests", {}).get("unit", {})
        if not unit:
            continue
        matrix.append(
            {
                "device": device,
                "include": unit.get("include", "*"),
                "exclude": unit.get("exclude", []),
            }
        )
    return matrix


def set_output(name: str, value: str) -> None:
    """Write a key=value pair to $GITHUB_OUTPUT (or print for local runs)."""
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"{name}<<EOF\n{value}\nEOF\n")
    else:
        print(f"{name}={value}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate CI test matrices")
    parser.add_argument(
        "--platform",
        required=True,
        help="Platform name (must match tests/platforms/<platform>.yaml)",
    )
    args = parser.parse_args(argv)

    config = load_platform(args.platform)
    devices = get_device_sections(config)
    unsupported = config.get("unsupported_features", [])

    if not devices:
        print(f"::warning::No device sections found in {args.platform}.yaml")

    e2e_matrix = build_e2e_matrix(config, devices, unsupported)
    unit_matrix = build_unit_matrix(config, devices)

    e2e_json = json.dumps(e2e_matrix, separators=(",", ":"))
    unit_json = json.dumps(unit_matrix, separators=(",", ":"))

    set_output("e2e", e2e_json)
    set_output("unit", unit_json)

    # Human-readable summary for CI logs
    print(f"Platform:    {args.platform}")
    print(f"Devices:     {devices}")
    print(f"E2E:         {len(e2e_matrix)} test(s)")
    for entry in e2e_matrix:
        print(
            f"  - {entry['task']}/{entry['model']}/{entry['case']} "
            f"(device={entry['device']}, timeout={entry['timeout']}m)"
        )
    print(f"Unit:        {len(unit_matrix)} config(s)")
    for entry in unit_matrix:
        print(
            f"  - device={entry['device']} "
            f"include={entry['include']} exclude={entry['exclude']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

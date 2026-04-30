# Unified Naming Convention: Vendor / Platform / Backend / Device / Kernel

## 1. Terminology Comparison Across Frameworks

### 1.1 PyTorch

| Concept | Definition | Examples |
|---------|-----------|----------|
| **device_type** | Type string of `torch.device`; the primary routing key for the dispatcher | `"cuda"`, `"cpu"`, `"xpu"`, `"mps"`, `"privateuseone"` |
| **backend** | Overloaded term: (1) dispatcher backend, synonymous with dispatch_key; (2) library configuration under `torch.backends.*` (e.g., cuDNN, MKL) | Dispatch: `"CUDA"`, `"CPU"`, `"XLA"` / Library: `torch.backends.cudnn` |
| **dispatch_key** | Internal routing key of the PyTorch dispatcher; finer-grained than device_type | `"CUDA"`, `"CPU"`, `"XLA"`, `"PrivateUse1"` |
| **device** | A `torch.device` object comprising device_type and an optional device index | `torch.device("cuda:0")`, `torch.device("npu:0")` |

PyTorch has no native concept of *vendor* or *platform*. Third-party hardware integrates via the `PrivateUse1` mechanism and may rename the device type to a custom string (e.g., `"npu"`, `"musa"`).

### 1.2 vLLM Upstream

| Concept | Definition | Examples |
|---------|-----------|----------|
| **Platform** | Abstract base class encapsulating device detection, capability queries, and runtime configuration | `CudaPlatform`, `RocmPlatform`, `TpuPlatform` |
| **PlatformEnum** | Coarse-grained enumeration of supported platforms | `CUDA`, `ROCM`, `TPU`, `XPU`, `CPU`, `OOT` |
| **device_type** | String passed to `torch.device()`; determines PyTorch dispatcher routing | CUDA → `"cuda"`, ROCm → `"cuda"`, TPU → `"tpu"` |
| **device_name** | vLLM-internal identifier used for configuration selection and logging | CUDA → `"cuda"`, ROCm → `"rocm"`, TPU → `"tpu"` |

Key observation: **ROCm sets device_type to `"cuda"`** (because it reuses PyTorch's CUDA interface), while its device_name is `"rocm"`. This demonstrates that device_type is a PyTorch runtime concept, whereas device_name is a vLLM-internal discriminator.

### 1.3 This Project (vllm-plugin-Fl)

The current `VENDOR_DEVICE_MAP` (`vllm_fl/utils.py:29-40`) is keyed by vendor_name and maps to device_type and device_name:

```python
VENDOR_DEVICE_MAP = {
    "nvidia":   {"device_type": "cuda", "device_name": "nvidia"},
    "ascend":   {"device_type": "npu",  "device_name": "npu"},
    "iluvatar": {"device_type": "cuda", "device_name": "cuda"},
    "metax":    {"device_type": "cuda", "device_name": "metax"},
    "mthreads": {"device_type": "musa", "device_name": "musa"},
}
```

CI configurations (`.github/configs/`) are keyed by platform:

| Config File | platform | Vendor in runner_labels |
|-------------|----------|------------------------|
| cuda.yml | cuda | nvidia |
| ascend.yml | ascend | ascend |
| maca.yml | maca | metax |
| musa.yml | musa | mthreads |

Test configurations (e.g., `tests/platforms/maca.yaml`) carry both platform and vendor fields:

```yaml
platform: maca
vendor: metax
```

---

## 2. Precise Definitions of the Five Core Concepts

### 2.1 Concept Hierarchy

```
┌──────────────────────────────────────────────────────────────────┐
│  Vendor                                                          │
│  "Who manufactured the hardware"                                 │
│  Top-level discriminator; determines both Platform and Backend   │
│  e.g., nvidia, ascend, metax, mthreads, iluvatar                 │
├──────────────────────────────┬───────────────────────────────────┤
│  Platform (Software Stack)   │  Backend                          │
│  "What software ecosystem    │  "What code implements the ops"   │
│   is required to run"        │  Determined by Vendor,            │
│  Determined by Vendor        │  NOT by Platform                  │
│  User-facing / CI / config   │  Operator dispatch / kernel route │
│  e.g., cuda, maca, musa      │  e.g., CudaBackend, MetaxBackend  │
│           │                  │           │                       │
│           ▼                  │           ▼                       │
│  Device                      │  Kernel                           │
│  device_type (PyTorch layer) │  Concrete op implementation       │
│  device_name (vLLM layer)    │  e.g., silu_and_mul_cuda          │
└──────────────────────────────┴───────────────────────────────────┘
```

> Note: Strictly speaking, CUDA, MXMACA, and MUSA are software stacks rather than platforms. However, the vLLM ecosystem — including upstream — conventionally uses them as platform identifiers to denote the required runtime environment.

### 2.2 Relationships Between Concepts

```
                    ┌──→ Platform (1) ──→ Device (1)
                    │                      ├── device_type (PyTorch layer)
                    │                      └── device_name (vLLM layer)
Vendor (1) ─────────┤
                    │
                    └──→ Backend (1..N) ──→ Kernel (N)
                          ├── vendor backend  (vendor-native)
                          ├── flaggems backend (Triton-based)
                          └── reference backend (pure PyTorch)
```

**Backend is associated directly with Vendor, not with Platform.** The rationale:

- Multiple vendors may share the same platform-level device_type (e.g., nvidia, metax, and iluvatar all report `device_type="cuda"`), yet each requires distinct kernel implementations.
- In this project, backend directories are organized by vendor: `dispatch/backends/vendor/<vendor_name>/`.
- The `Backend` base class exposes a `vendor` property — not a `platform` property.
- Backend availability checks verify vendor identity (e.g., `device_name == "nvidia"`), not platform membership.

Platform operates at a higher level of abstraction: it determines device_type, dispatch_key, and dist_backend for runtime configuration. Backend operates at a lower level: it determines which set of kernel implementations to invoke.

### 2.3 Detailed Concept Differentiation

#### Vendor vs. Platform

- **Vendor** answers *"who manufactured the hardware"*: nvidia, ascend (Huawei), metax (MetaX), mthreads (Moore Threads), iluvatar (Iluvatar CoreX).
- **Platform** answers *"what software stack is required to run"*: cuda, ascend, maca (MXMACA), musa.
- The mapping is typically one-to-one but not necessarily so. In principle, NVIDIA hardware could run under either the CUDA or ROCm software stack.

> **On MACA**: MXMACA® is MetaX's official software stack brand, analogous to NVIDIA's CUDA. It is formally a software stack, not a platform. However, since vLLM upstream already treats CUDA as a platform identifier (`CudaPlatform`), using `platform: maca` follows the same established convention.

#### Vendor vs. Backend

- **Vendor** is the hardware manufacturer identifier; **Backend** is the collection of operator implementations for that vendor's hardware.
- Backend is determined directly by Vendor, not by Platform.
- Rationale: under the same platform (device_type=`"cuda"`), nvidia, metax, and iluvatar have different hardware architectures and require independent backend implementations.
- In this project, backend directories are organized by vendor: `dispatch/backends/vendor/<vendor_name>/`.
- The `Backend` base class defines a `vendor` property (not `platform`); e.g., `CudaBackend.vendor` returns `"nvidia"`.

#### device_type vs. device_name

- **device_type** is a PyTorch runtime concept — the string passed to `torch.device()` that determines dispatcher routing.
- **device_name** is a vLLM-internal concept — used to disambiguate vendors that share the same device_type.

The distinction is necessary because **device_type collisions occur**:

| Vendor | device_type | Reason |
|--------|-------------|--------|
| NVIDIA | cuda | Native CUDA |
| MetaX | cuda | MACA is CUDA-API compatible |
| Iluvatar | cuda | CUDA-API compatible |

All three report `device_type="cuda"` and `torch.cuda.is_available()` returns `True` for each. device_type alone cannot distinguish the vendor; vendor_name / device_name serves as the supplementary discriminator.

This also explains why backends must be associated with vendors rather than platforms — at the platform level, device_type cannot differentiate vendors; only vendor identity enables precise routing to the correct backend.

#### Backend vs. Kernel

- **Backend** is the organizational unit for operator implementations; a single backend contains multiple kernels.
- **Kernel** is the concrete implementation of a single operator on specific hardware.

This project organizes backends in three tiers:

```
dispatch/backends/
├── flaggems/      # FlagGems Triton kernels (cross-vendor; not tied to a specific vendor)
├── reference/     # Reference implementations (pure PyTorch; used for correctness validation)
└── vendor/        # Vendor-specific implementations (organized by vendor, not by platform)
    ├── cuda/      #   vendor: nvidia
    ├── ascend/    #   vendor: ascend (Huawei)
    ├── metax/     #   vendor: metax (MetaX)
    ├── mthreads/  #   vendor: mthreads (Moore Threads)
    └── iluvatar/  #   vendor: iluvatar (Iluvatar CoreX)
```

#### dispatch_key

The dispatch_key is the internal routing key of the PyTorch dispatcher, determining which code path an operator invocation follows:

| dispatch_key | Semantics |
|-------------|-----------|
| `"CUDA"` | Routes to CUDA-registered kernels (shared by NVIDIA, MetaX, and Iluvatar) |
| `"PrivateUse1"` | Routes to third-party registered kernels (Ascend, Moore Threads) |
| `"CPU"` | Routes to CPU kernels |
| `"XLA"` | Routes to XLA compilation path (TPU) |

---

## 3. Complete Vendor Mapping Table

| Vendor | Platform | device_type | device_name | dispatch_key | dist_backend | vendor_backend_dir | cuda_alike |
|--------|----------|-------------|-------------|--------------|-------------|-------------------|------------|
| nvidia | cuda | cuda | nvidia | CUDA | nccl | cuda | Yes |
| ascend | ascend | npu | npu | PrivateUse1 | hccl | ascend | No |
| metax | maca | cuda | metax | CUDA | nccl | metax | Yes |
| mthreads | musa | musa | musa | PrivateUse1 | mccl | mthreads | Yes* |
| iluvatar | iluvatar | cuda | cuda | CUDA | nccl | iluvatar | No* |

**Notes:**
- *mthreads*: device_type is not `"cuda"`, but `is_cuda_alike()` returns `True` in this project because MUSA is semantically compatible with CUDA operations.
- *iluvatar*: device_type is `"cuda"`, but `is_cuda_alike()` returns `False` due to behavioral differences that require special handling.

---

## 4. Proposed Unified Data Structure

### HardwareProfile

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareProfile:
    """
    Immutable descriptor consolidating all hardware-specific identifiers
    and configuration for a given vendor.

    Fields:
        vendor:             Canonical vendor identifier (primary key).
        platform:           Software stack / runtime environment identifier.
        device_type:        String for torch.device() construction.
        dispatch_key:       PyTorch dispatcher routing key.
        device_name:        vLLM-internal device identifier.
        cuda_alike:         Whether the device is semantically CUDA-compatible.
        dist_backend:       Distributed communication backend.
        vendor_backend_dir: Subdirectory name under dispatch/backends/vendor/.
    """
    vendor: str
    platform: str
    device_type: str
    dispatch_key: str
    device_name: str
    cuda_alike: bool
    dist_backend: str
    vendor_backend_dir: str
```

### Hardware Registry

```python
HARDWARE_REGISTRY: dict[str, HardwareProfile] = {
    "nvidia": HardwareProfile(
        vendor="nvidia",
        platform="cuda",
        device_type="cuda",
        dispatch_key="CUDA",
        device_name="nvidia",
        cuda_alike=True,
        dist_backend="nccl",
        vendor_backend_dir="cuda",
    ),
    "ascend": HardwareProfile(
        vendor="ascend",
        platform="ascend",
        device_type="npu",
        dispatch_key="PrivateUse1",
        device_name="npu",
        cuda_alike=False,
        dist_backend="hccl",
        vendor_backend_dir="ascend",
    ),
    "metax": HardwareProfile(
        vendor="metax",
        platform="maca",
        device_type="cuda",
        dispatch_key="CUDA",
        device_name="metax",
        cuda_alike=True,
        dist_backend="nccl",
        vendor_backend_dir="metax",
    ),
    "mthreads": HardwareProfile(
        vendor="mthreads",
        platform="musa",
        device_type="musa",
        dispatch_key="PrivateUse1",
        device_name="musa",
        cuda_alike=True,
        dist_backend="mccl",
        vendor_backend_dir="mthreads",
    ),
    "iluvatar": HardwareProfile(
        vendor="iluvatar",
        platform="iluvatar",
        device_type="cuda",
        dispatch_key="CUDA",
        device_name="cuda",
        cuda_alike=False,
        dist_backend="nccl",
        vendor_backend_dir="iluvatar",
    ),
}
```

### Advantages of This Structure

- **Vendor as the unique key** eliminates ambiguity caused by device_type collisions.
- **All mappings consolidated in a single location**, replacing scattered definitions across `VENDOR_DEVICE_MAP`, `dist_backend_dict`, and `is_cuda_alike()`.
- **Adding a new vendor requires a single registry entry** rather than modifications across multiple files.

### Query Interface

```python
def get_profile_by_vendor(vendor: str) -> HardwareProfile:
    """Look up a hardware profile by vendor name (primary entry point)."""
    return HARDWARE_REGISTRY[vendor]


def get_profile_by_platform(platform: str) -> HardwareProfile:
    """Look up a hardware profile by platform name (used in CI/config contexts)."""
    for profile in HARDWARE_REGISTRY.values():
        if profile.platform == platform:
            return profile
    raise ValueError(f"Unknown platform: {platform}")
```

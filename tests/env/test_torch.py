"""Environment baseline for torch (KOM-020a): the trainer's dependency exists.

These are environment assertions, not library tests: they pin what the nix
devshell hands us, so a silent change in the pinned nixpkgs (a torch version
bump, a CPU/GPU build swap) fails here rather than deep inside the trainer.

The GPU tests below skip when no accelerator is visible (KOM-055): the default
devshell is the ROCm torch build, which sees no device on this CUDA-only host.
Campaigns and benches select `.#cuda` explicitly and enforce the device at
runtime, so skipping here cannot hide a real regression there.
"""

from __future__ import annotations

import os

import pytest
import torch

# True only when torch can actually reach an accelerator. The default (ROCm)
# shell on a CUDA-only box reports no device, which is what gates the skips.
_ACCELERATOR_VISIBLE: bool = torch.cuda.is_available() and torch.cuda.device_count() >= 1

_NO_ACCELERATOR_REASON = (
    "KOM-055: no accelerator visible -- the default ROCm devshell sees no device on a "
    "CUDA-only host; campaigns select `.#cuda` and enforce the device at runtime."
)

# The pinned nixpkgs (867dcbc30bafe3c862ef88620f2e7a109d7d3be5) builds
# python312Packages.torch at exactly this version. Bumping the pin is a
# deliberate act; this assertion makes it a visible one.
#
# KOM-1221 (D15): `.#test` exports KOMUGI_EXPECTED_TORCH_VERSION as its own
# nixpkgs-test torch's real version, so this shell's CPU-only torch (which
# tracks a different channel) is checked against itself rather than the
# training pin below. The training shells (default/cuda/rocm) never set the
# variable, so this still enforces 2.12.0 there.
_EXPECTED_TORCH_VERSION = os.environ.get("KOMUGI_EXPECTED_TORCH_VERSION", "2.12.0")


def test_torch_version_matches_the_pin() -> None:
    """torch imports, and it is the version the flake pin promises."""
    # torch.__version__ is a TorchVersion (a str subclass); compare as text.
    assert str(torch.__version__) == _EXPECTED_TORCH_VERSION


def test_cpu_float32_matmul_is_exact() -> None:
    """A 2x2 float32 matmul on CPU produces the exact expected values.

    The operands and the products are all small integers, so float32 holds
    them exactly -- this is an equality check by construction, not a
    tolerance check that happens to pass.
    """
    left = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, device="cpu")
    right = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32, device="cpu")
    expected = torch.tensor([[19.0, 22.0], [43.0, 50.0]], dtype=torch.float32, device="cpu")

    product = left @ right

    assert product.dtype == torch.float32
    assert product.device.type == "cpu"
    assert product.shape == expected.shape
    assert torch.equal(product, expected)


@pytest.mark.skipif(not _ACCELERATOR_VISIBLE, reason=_NO_ACCELERATOR_REASON)
def test_gpu_is_available() -> None:
    """A GPU device is present and usable from this devshell (KOM-020b).

    The flake installs the ROCm build of torch and the shellHook exports
    HSA_OVERRIDE_GFX_VERSION=10.3.0 for the gfx1034 card. Asserting presence
    keeps that wiring honest: if the closure ever falls back to a CPU-only
    torch, or the HSA override stops taking, this fails loudly here instead of
    the trainer silently running an order of magnitude slower on CPU.
    """
    assert torch.cuda.is_available() is True
    assert torch.cuda.device_count() >= 1


@pytest.mark.skipif(not _ACCELERATOR_VISIBLE, reason=_NO_ACCELERATOR_REASON)
def test_gpu_float32_matmul_matches_cpu() -> None:
    """The same 2x2 float32 matmul agrees exactly between GPU and CPU.

    Operands and products are small integers, so float32 holds them exactly on
    either device -- this is an equality check by construction, not a tolerance
    check that happens to pass. It exercises the device end to end: allocation,
    kernel dispatch, and the copy back.
    """
    left = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    right = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32)
    expected = left @ right

    product = (left.to("cuda") @ right.to("cuda")).cpu()

    assert product.dtype == torch.float32
    assert product.shape == expected.shape
    assert torch.equal(product, expected)


def test_torch_backend_is_exactly_one_of_hip_or_cuda() -> None:
    """torch is an accelerator build: HIP xor CUDA, never both and never neither.

    torch keeps the `torch.cuda` namespace for its ROCm builds, so device
    availability alone cannot tell the two apart. The default devshell installs
    the ROCm build, which sets `torch.version.hip`; the `.#cuda` devshell
    installs the NVIDIA build, which sets `torch.version.cuda`. A CPU-only
    closure sets neither and fails here -- which is the point, since that
    fallback is invisible until the trainer runs an order of magnitude slower.

    KOM-1221 (D15): `.#test` (and `.#ci`) export KOMUGI_EXPECTED_TORCH_BACKEND
    = "cpu" because their torch really is a CPU-only build -- there, neither
    hip nor cuda is the expected shape, not a fallback. The training shells
    never set the variable, so they keep asserting hip xor cuda as today.
    """
    hip_version: object = torch.version.hip
    cuda_version: object = torch.version.cuda

    has_hip = isinstance(hip_version, str) and hip_version != ""
    has_cuda = isinstance(cuda_version, str) and cuda_version != ""

    if os.environ.get("KOMUGI_EXPECTED_TORCH_BACKEND") == "cpu":
        assert not has_hip and not has_cuda, (
            f"expected a CPU-only torch build, got hip={hip_version!r} cuda={cuda_version!r}"
        )
    else:
        assert has_hip != has_cuda, (
            f"expected exactly one accelerator backend, got hip={hip_version!r} cuda={cuda_version!r}"
        )

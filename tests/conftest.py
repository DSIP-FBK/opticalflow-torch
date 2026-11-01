"""Shared fixtures. Tests marked ``cuda`` (including the ``cuda`` device parameter) are skipped without CUDA.

Tests marked ``compile`` are skipped before PyTorch 2.7, whose Inductor miscompiles or crashes on this package.
"""

from __future__ import annotations

import pytest
import torch

DEVICES = [pytest.param("cpu", id="cpu"), pytest.param("cuda", id="cuda", marks=pytest.mark.cuda)]
COMPILE_MINIMUM = (2, 7)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "compile: uses torch.compile, skipped before PyTorch 2.7")
    # emitted by Triton when torch.compile runs on CUDA; unrelated to this package
    config.addinivalue_line("filterwarnings", "ignore:TRITON_DEFAULT_FP_FUSION:UserWarning")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    skips = {}
    if not torch.cuda.is_available():
        skips["cuda"] = pytest.mark.skip(reason="CUDA is not available")
    if tuple(int(part) for part in torch.__version__.split(".")[:2]) < COMPILE_MINIMUM:
        skips["compile"] = pytest.mark.skip(reason=f"torch.compile needs PyTorch 2.7 or later, got {torch.__version__}")
    for item in items:
        for marker, skip in skips.items():
            if item.get_closest_marker(marker) is not None:
                item.add_marker(skip)


@pytest.fixture(params=DEVICES)
def device(request: pytest.FixtureRequest) -> torch.device:
    """Each available device."""
    return torch.device(request.param)


@pytest.fixture(params=[torch.float32, torch.float64], ids=["float32", "float64"])
def dtype(request: pytest.FixtureRequest) -> torch.dtype:
    """Each supported floating-point dtype."""
    return request.param


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)


@pytest.fixture
def grad_scaler() -> torch.amp.GradScaler:
    """A CUDA gradient scaler; ``torch.amp.GradScaler`` is ``torch.cuda.amp.GradScaler`` before PyTorch 2.3."""
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda")
    return torch.cuda.amp.GradScaler()

"""Tests of the Farneback optical flow, mostly against OpenCV's ``calcOpticalFlowFarneback``."""

from __future__ import annotations

import importlib
import math

import numpy as np
import pytest
import torch
from synthetic import (
    AnalyticTexture,
    flow_to_cv,
    image_to_cv,
    make_pair,
    rotation_flow,
    smooth_texture,
    translation_flow,
    zoom_flow,
)
from torch.autograd import gradcheck

from opticalflow_torch import Farneback, farneback
from opticalflow_torch._ops import pyramid_level, pyramid_levels, separable_filter

cv2 = pytest.importorskip("cv2")
fb = importlib.import_module("opticalflow_torch.farneback")  # the package exports a function of the same name

DEFAULTS = dict(pyr_scale=0.5, levels=5, winsize=13, iterations=10, poly_n=5, poly_sigma=1.1)
LIVEMAP = dict(pyr_scale=0.5, levels=3, winsize=25, iterations=3, poly_n=5, poly_sigma=1.2)  # meteotn-bot livemap

# 197x253 is odd and non-square. pyr_scale 0.7 lets it hold all 5 coarser levels (0.5 stops at 2).
PARITY_SIZE = (197, 253)
PARITY_CASES = {
    "defaults": DEFAULTS,
    "livemap": LIVEMAP,
    "gaussian": dict(DEFAULTS, window="gaussian"),
    "poly_n7": dict(DEFAULTS, poly_n=7, poly_sigma=1.5),
    "pyr_scale0.8": dict(DEFAULTS, pyr_scale=0.8),
    "winsize12": dict(DEFAULTS, winsize=12),
    **{f"levels{k}": dict(DEFAULTS, pyr_scale=0.7, levels=k) for k in range(6)},
    # one iteration on a small window, so that the result still depends on the initial flow
    "initial_flow": dict(DEFAULTS, levels=2, iterations=1, winsize=5),
}


def cv_farneback(
    frame0: torch.Tensor, frame1: torch.Tensor, params: dict, initial_flow: torch.Tensor | None = None
) -> np.ndarray:
    flags = cv2.OPTFLOW_FARNEBACK_GAUSSIAN if params.get("window") == "gaussian" else 0
    flow = None
    if initial_flow is not None:
        flags |= cv2.OPTFLOW_USE_INITIAL_FLOW
        flow = flow_to_cv(initial_flow).copy()
    names = ("pyr_scale", "levels", "winsize", "iterations", "poly_n", "poly_sigma")
    return cv2.calcOpticalFlowFarneback(
        image_to_cv(frame0), image_to_cv(frame1), flow, *(params[name] for name in names), flags
    )


def opencv_inside(coord: torch.Tensor, size: int) -> torch.Tensor:
    """OpenCV's hard test, to swap for the smooth ``_inside`` weight."""
    start = coord.floor()
    return ((start >= 0) & (start < size - 1)).to(coord.dtype)


def parity_scene(device: torch.device | str = "cpu") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    height, width = PARITY_SIZE
    true = rotation_flow(height, width, 2.0) + translation_flow(height, width, 1.3, -0.8)
    frame0, frame1 = make_pair(AnalyticTexture(11), true)
    initial = 0.8 * true + 0.3
    return frame0.to(device), frame1.to(device), initial.to(device)


def parity_initial_flow(device: torch.device | str = "cpu") -> torch.Tensor:
    """Initial flow of the "initial_flow" case, non-affine so that INTER_AREA and INTER_LINEAR shrink it apart."""
    _, _, initial = parity_scene(device)
    noise = smooth_texture(*PARITY_SIZE, channels=2, seed=7, sigma=3.0, device=device)
    return initial + (noise / 255 - 0.5) * 8


def run_parity(case: str, device: torch.device) -> np.ndarray:
    """Endpoint differences to OpenCV, shape (H, W)."""
    params = PARITY_CASES[case]
    frame0, frame1, _ = parity_scene(device)
    initial = parity_initial_flow(device) if case == "initial_flow" else None
    expected = cv_farneback(frame0, frame1, params, initial)
    out = flow_to_cv(farneback(frame0, frame1, initial_flow=initial, **params)).astype(np.float64)
    return np.hypot(*(out - expected).transpose(2, 0, 1))


def border_distance(height: int, width: int) -> np.ndarray:
    y, x = np.mgrid[:height, :width]
    return np.minimum(np.minimum(y, height - 1 - y), np.minimum(x, width - 1 - x))


def summary(errors: np.ndarray) -> tuple[float, float, float]:
    return float(errors.mean()), float(np.percentile(errors, 99)), float(errors.max())


# ----------------------------------------------------------------------------------------------
# Parity with OpenCV
# ----------------------------------------------------------------------------------------------


def test_parity_cases_use_distinct_levels() -> None:
    for k in range(6):
        assert len(pyramid_levels(*PARITY_SIZE, 0.7, k)) == k + 1


def test_initial_flow_case_depends_on_the_initial_flow() -> None:
    # otherwise its parity test would not notice a port that ignores or mis-resizes initial_flow
    params = PARITY_CASES["initial_flow"]
    frame0, frame1, _ = parity_scene()
    used = cv_farneback(frame0, frame1, params, parity_initial_flow())
    ignored = cv_farneback(frame0, frame1, params)
    inner = border_distance(*PARITY_SIZE) >= PARITY_BAND
    assert np.hypot(*(used - ignored).transpose(2, 0, 1))[inner].mean() > 5e-4


@pytest.mark.parametrize("case", list(PARITY_CASES))
def test_matches_opencv_with_its_inside_test(case: str, device: torch.device, monkeypatch: pytest.MonkeyPatch) -> None:
    # With OpenCV's hard inside test the port is exact up to float32 rounding, borders included.
    # Measured over the cases (flows up to 7 px): mean <= 1.2e-6, p99 <= 6.2e-6, max <= 7.3e-5.
    monkeypatch.setattr(fb, "_inside", opencv_inside)
    errors = run_parity(case, device)
    mean, p99, worst = summary(errors)
    print(f"{case} (hard test), whole frame: mean {mean:.1e} p99 {p99:.1e} max {worst:.1e} px")
    assert mean < 3e-6 and p99 < 2e-5 and worst < 2e-4


# The smooth inside weight changes the equations of pixels whose displaced position is within a
# pixel of leaving the image, and the window averaging carries that inwards by up to winsize // 2
# pixels per iteration; beyond 32 pixels every case is back to rounding. Measured over the cases:
#   interior        mean <= 1.1e-6, p99 <= 5.0e-6, max <= 4.1e-5
#   whole, box      mean <= 3.3e-3, p99 <= 7.1e-2, max <= 0.99
#   whole, gaussian mean    1.3e-2, p99    3.0e-1, max    7.7 (OpenCV's border flow reaches 18 px)
#   whole, initial_flow mean 1.2e-2, p99 3.4e-1, max 3.4 (single iteration from a +-4 px start)
PARITY_BAND = 32
WHOLE_FRAME_TOLERANCES = {"box": (5e-3, 0.1, 1.5), "gaussian": (2e-2, 0.5, 10.0), "initial_flow": (2e-2, 0.5, 5.0)}


@pytest.mark.parametrize("case", list(PARITY_CASES))
def test_matches_opencv(case: str, device: torch.device) -> None:
    errors = run_parity(case, device)
    inner = summary(errors[border_distance(*PARITY_SIZE) >= PARITY_BAND])
    whole = summary(errors)
    print(
        f"{case}: interior mean {inner[0]:.1e} p99 {inner[1]:.1e} max {inner[2]:.1e} | "
        f"whole frame mean {whole[0]:.1e} p99 {whole[1]:.1e} max {whole[2]:.1e} px"
    )
    assert inner[0] < 3e-6 and inner[1] < 2e-5 and inner[2] < 1e-4
    tolerances = WHOLE_FRAME_TOLERANCES.get(case) or WHOLE_FRAME_TOLERANCES[PARITY_CASES[case].get("window", "box")]
    assert all(value < tol for value, tol in zip(whole, tolerances, strict=True))


@pytest.mark.parametrize("size", [(7, 9), (9, 7), (5, 12), (1, 17), (17, 1), (2, 2), (11, 3), (40, 70)])
@pytest.mark.parametrize("window", ["box", "gaussian"])
def test_matches_opencv_on_small_frames(size: tuple[int, int], window: str, monkeypatch: pytest.MonkeyPatch) -> None:
    # below 10 pixels OpenCV's unsigned border test picks unexpected pixels, reproduced here
    monkeypatch.setattr(fb, "_inside", opencv_inside)
    frame0 = smooth_texture(*size, seed=1, sigma=1.5)
    frame1 = 0.7 * frame0 + 0.3 * smooth_texture(*size, seed=2, sigma=1.5)
    params = dict(DEFAULTS, window=window)
    expected = cv_farneback(frame0, frame1, params)
    out = flow_to_cv(farneback(frame0, frame1, **params))
    assert np.abs(out - expected).max() < 1e-4 * max(1.0, np.abs(expected).max())


def test_flow_is_continuous_where_samples_leave_the_image(monkeypatch: pytest.MonkeyPatch) -> None:
    # A uniform initial x-flow of 1 -+ d moves the samples of the second-to-last column across
    # OpenCV's inside test, where its flow jumps; the smooth weight makes the change O(d).
    height, width = 23, 29
    frame0, frame1 = small_pair(height, width)

    def change(d: float) -> float:
        flows = []
        for shift in (1 - d, 1 + d):
            initial = torch.zeros(1, 2, height, width, dtype=torch.float64)
            initial[:, 0] = shift
            flows.append(farneback(frame0, frame1, initial_flow=initial, levels=0, winsize=5, iterations=1))
        return float((flows[1] - flows[0]).abs().max())

    assert change(1e-7) < 1e-5
    monkeypatch.setattr(fb, "_inside", opencv_inside)
    assert change(1e-7) > 0.1  # the jump that the smooth weight removes


def test_default_window_sigma() -> None:
    frame0, frame1, _ = parity_scene()
    kwargs = dict(levels=1, iterations=2, winsize=15, window="gaussian")
    expected = farneback(frame0, frame1, window_sigma=0.3 * 7, **kwargs)
    torch.testing.assert_close(farneback(frame0, frame1, **kwargs), expected, rtol=0, atol=0)


def test_winsize1_box_is_the_single_pixel_system() -> None:
    # OpenCV's running sums count row 0 and column 0 again for winsize 1 (a slip not reproduced)
    frame0, frame1, _ = parity_scene()
    box = farneback(frame0, frame1, winsize=1, levels=1, iterations=2)
    single = farneback(frame0, frame1, winsize=1, levels=1, iterations=2, window="gaussian")
    torch.testing.assert_close(box, single, rtol=0, atol=0)


# ----------------------------------------------------------------------------------------------
# Polynomial expansion
# ----------------------------------------------------------------------------------------------


def expansion_kernels(n: int, sigma: float) -> np.ndarray:
    """2-D kernels of ``(ry, rx, ryy, rxx, rxy)`` from a direct inverse of OpenCV's Gram matrix."""
    t = np.arange(-n, n + 1.0)
    g = np.exp(-t * t / (2 * sigma * sigma))
    g /= g.sum()
    y, x = np.meshgrid(t, t, indexing="ij")
    weights = np.outer(g, g)
    basis = np.stack((np.ones_like(x), x, y, x * x, y * y, x * y))
    inverse = np.linalg.inv(np.einsum("iyx,jyx,yx->ij", basis, basis, weights))
    return np.einsum("ri,iyx,yx->ryx", inverse[[2, 1, 4, 3, 5]], basis, weights)


def port_kernels(vertical: torch.Tensor, horizontal: torch.Tensor) -> torch.Tensor:
    """2-D kernels of ``(ry, rx, ryy, rxx, rxy)`` from the port's separable ones."""
    return torch.stack([torch.outer(vertical[v], horizontal[h]) for h, v in enumerate((1, 0, 2, 0, 1))])


@pytest.mark.parametrize("n", [1, 5, 7])
@pytest.mark.parametrize("sigma", [0.17, 1.1, 10.0])
def test_expansion_kernels_match_opencvs_gram_matrix(n: int, sigma: float) -> None:
    expected = expansion_kernels(n, sigma)
    out = port_kernels(*fb._poly_kernels(n, sigma, torch.empty(0, dtype=torch.float64))).numpy()
    np.testing.assert_allclose(out, expected, rtol=0, atol=1e-12 * np.abs(expected).max())


def test_small_poly_sigma_tends_to_central_differences(dtype: torch.dtype) -> None:
    # Below about 0.164 OpenCV fails to invert its Gram matrix (and returns a zero flow); the
    # port's kernels stay finite and tend to central differences.
    delta, first, second = torch.zeros(3, 11, dtype=dtype)
    delta[5] = 1
    first[4], first[6] = -0.5, 0.5
    second[4:7] = torch.tensor([0.5, -1.0, 0.5])
    expected = port_kernels(torch.stack((delta, first, second)), torch.stack((delta, first, delta, second, first)))
    for value in (1.2e-7, 1e-3, 0.05, 0.12):
        sigma = torch.tensor(value, dtype=dtype, requires_grad=True)
        kernels = fb._poly_kernels(5, sigma, torch.empty(0, dtype=dtype))
        torch.testing.assert_close(port_kernels(*kernels), expected, rtol=0, atol=1e-6)
        (grad,) = torch.autograd.grad(sum(k.square().sum() for k in kernels), sigma)
        assert torch.isfinite(grad)


def test_small_poly_sigma_gives_a_finite_flow(device: torch.device, dtype: torch.dtype) -> None:
    true = translation_flow(96, 128, 1.5, -0.7, device=device, dtype=dtype)
    frame0, frame1 = make_pair(AnalyticTexture(9), true)
    for value in (0.03, 0.12):
        sigma = torch.tensor(value, device=device, dtype=dtype, requires_grad=True)
        flow = farneback(frame0, frame1, poly_sigma=sigma)
        (grad,) = torch.autograd.grad(flow.square().mean(), sigma)
        errors = (flow - true).norm(dim=1)[:, 16:-16, 16:-16]
        assert torch.isfinite(grad) and errors.mean() < 0.02 and errors.max() < 0.1
        assert not cv_farneback(frame0, frame1, dict(DEFAULTS, poly_sigma=value)).any()


# ----------------------------------------------------------------------------------------------
# Accuracy on exact motions
# ----------------------------------------------------------------------------------------------

ACCURACY_SIZE = (160, 192)
# interior EPE (24-pixel band excluded) measured with the defaults: mean / max
# translation 0.012 / 0.068, large translation 0.011 / 0.072, rotation 0.113 / 0.379, zoom 0.045 / 0.157
MOTIONS = {
    "translation": (translation_flow(*ACCURACY_SIZE, 0.4, -0.7), 0.02, 0.1),
    "large translation": (translation_flow(*ACCURACY_SIZE, 5.5, -4.2), 0.02, 0.1),
    "rotation": (rotation_flow(*ACCURACY_SIZE, 3.0), 0.15, 0.5),
    "zoom": (zoom_flow(*ACCURACY_SIZE, 1.03), 0.07, 0.25),
}


@pytest.mark.parametrize("motion", list(MOTIONS))
def test_accuracy_on_exact_motions(motion: str, device: torch.device) -> None:
    true, mean_tol, max_tol = MOTIONS[motion]
    frame0, frame1 = make_pair(AnalyticTexture(2), true.to(device))
    errors = (farneback(frame0, frame1) - true.to(device)).norm(dim=1)[:, 24:-24, 24:-24]
    assert errors.mean() < mean_tol and errors.max() < max_tol


# ----------------------------------------------------------------------------------------------
# Gradients
# ----------------------------------------------------------------------------------------------


def small_pair(height: int, width: int, dtype: torch.dtype = torch.float64) -> tuple[torch.Tensor, torch.Tensor]:
    true = translation_flow(height, width, 0.6, -0.3, dtype=dtype)
    return make_pair(AnalyticTexture(4), true)


def test_gradcheck_frames_and_initial_flow() -> None:
    frame0, frame1 = small_pair(9, 11)
    initial = (0.6 * torch.rand(1, 2, 9, 11, dtype=torch.float64) + 0.2).requires_grad_()
    frame0.requires_grad_()
    frame1.requires_grad_()

    def fn(a: torch.Tensor, b: torch.Tensor, flow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return farneback(a, b, levels=0, winsize=5, iterations=2, initial_flow=flow, return_confidence=True)

    assert gradcheck(fn, (frame0, frame1, initial))


def test_gradcheck_with_pyramid() -> None:
    frame0, frame1 = small_pair(40, 44)
    initial = (0.6 * torch.rand(1, 2, 40, 44, dtype=torch.float64) + 0.2).requires_grad_()
    frame0.requires_grad_()
    frame1.requires_grad_()

    def fn(a: torch.Tensor, b: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        return farneback(a, b, pyr_scale=0.8, levels=1, winsize=7, iterations=2, initial_flow=flow)

    assert len(pyramid_levels(40, 44, 0.8, 1)) == 2
    assert gradcheck(fn, (frame0, frame1, initial), fast_mode=True)


@pytest.mark.parametrize("window", ["box", "gaussian"])
def test_gradcheck_parameters(window: str) -> None:
    frame0, frame1 = small_pair(12, 13)
    poly_sigma = torch.tensor(1.2, dtype=torch.float64, requires_grad=True)
    window_sigma = torch.tensor(1.4, dtype=torch.float64, requires_grad=True)
    eps = torch.tensor(5.0, dtype=torch.float64, requires_grad=True)

    def fn(
        poly_sigma: torch.Tensor, window_sigma: torch.Tensor, eps: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return farneback(
            frame0,
            frame1,
            levels=0,
            winsize=7,
            iterations=2,
            poly_sigma=poly_sigma,
            window=window,
            window_sigma=window_sigma if window == "gaussian" else None,
            eps=eps,
            return_confidence=True,
        )

    assert gradcheck(fn, (poly_sigma, window_sigma, eps))


@pytest.mark.parametrize("value", [0.0, 117.0])
def test_finite_gradients_on_constant_frames(value: float, device: torch.device) -> None:
    frame0 = torch.full((2, 1, 40, 36), value, device=device, requires_grad=True)
    frame1 = torch.full((2, 1, 40, 36), value, device=device, requires_grad=True)
    model = Farneback(window="gaussian", learnable=True, levels=2, iterations=3).to(device)
    flow, confidence = model(frame0, frame1, return_confidence=True)
    (flow.square().sum() + confidence.sum()).backward()
    assert torch.isfinite(flow).all() and torch.isfinite(confidence).all()
    grads = [frame0.grad, frame1.grad, *(p.grad for p in model.parameters())]
    assert all(torch.isfinite(g).all() for g in grads)


def test_finite_second_order_gradients_with_flat_regions(device: torch.device, dtype: torch.dtype) -> None:
    # a textured disc on an exactly zero background, like rain on dry ground in radar fields
    height = width = 64
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    disc = ((y - 32) ** 2 + (x - 32) ** 2 < 18**2).to(device, dtype)
    texture = smooth_texture(height, width, seed=3, device=device, dtype=dtype) * disc
    frame0 = texture.clone().requires_grad_()
    frame1 = torch.roll(texture, (1, 2), (-2, -1))
    model = Farneback(window="gaussian", levels=1, iterations=3, learnable=True).to(device, dtype)
    flow, confidence = model(frame0, frame1, return_confidence=True)
    assert (confidence[..., :8, :8] == 0).all()
    inputs = [frame0, *model.parameters()]
    grads = torch.autograd.grad(flow.square().sum() + confidence.mean(), inputs, create_graph=True)
    second = torch.autograd.grad(sum(g.square().sum() for g in grads), inputs)
    assert all(torch.isfinite(s).all() for s in second)


def test_poly_sigma_below_float_epsilon() -> None:
    """A number below float32's epsilon is OpenCV's default; a tensor is clamped, without a jump."""
    frame0, frame1, _ = parity_scene()
    kwargs = dict(levels=1, iterations=2)
    for value in (0.0, -1.0, 1e-8):
        torch.testing.assert_close(
            farneback(frame0, frame1, poly_sigma=value, **kwargs),
            farneback(frame0, frame1, poly_sigma=5 * 0.3, **kwargs),
        )
    at_epsilon = farneback(frame0, frame1, poly_sigma=torch.tensor(1.2e-7), **kwargs)
    for value in (0.0, -1.0, 1.18e-7):
        sigma = torch.tensor(value, requires_grad=True)
        out = farneback(frame0, frame1, poly_sigma=sigma, **kwargs)
        torch.testing.assert_close(out, at_epsilon)
        (grad,) = torch.autograd.grad(out.sum(), sigma)
        assert grad == 0


# ----------------------------------------------------------------------------------------------
# Batches, channels, dtypes and devices
# ----------------------------------------------------------------------------------------------


def test_batch_equals_per_sample(device: torch.device) -> None:
    height, width = 70, 90
    pairs = [make_pair(AnalyticTexture(s), translation_flow(height, width, 0.5 * s, -0.3 * s)) for s in range(3)]
    frame0 = torch.cat([p[0] for p in pairs]).to(device)
    frame1 = torch.cat([p[1] for p in pairs]).to(device)
    initial = torch.randn(3, 2, height, width, device=device)
    flow, confidence = farneback(frame0, frame1, initial_flow=initial, return_confidence=True)
    for i in range(3):
        f, c = farneback(frame0[i : i + 1], frame1[i : i + 1], initial_flow=initial[i : i + 1], return_confidence=True)
        torch.testing.assert_close(flow[i : i + 1], f, rtol=0, atol=1e-5)
        torch.testing.assert_close(confidence[i : i + 1], c, rtol=1e-6, atol=0)


def test_channels_sum_their_equations(device: torch.device) -> None:
    frame0, frame1, _ = parity_scene(device)
    other0 = smooth_texture(*PARITY_SIZE, seed=3, device=device)
    other1 = torch.roll(other0, (1, 2), (-2, -1))
    # duplicated channels double every term, i.e. the same as eps / 4 with one channel
    doubled = farneback(torch.cat((frame0, frame0), 1), torch.cat((frame1, frame1), 1), eps=4e-3)
    torch.testing.assert_close(doubled, farneback(frame0, frame1, eps=1e-3), rtol=0, atol=1e-5)
    # the order of the channels does not matter
    a = farneback(torch.cat((frame0, other0), 1), torch.cat((frame1, other1), 1))
    b = farneback(torch.cat((other0, frame0), 1), torch.cat((other1, frame1), 1))
    torch.testing.assert_close(a, b, rtol=0, atol=1e-4)


def test_dtypes(dtype: torch.dtype, device: torch.device) -> None:
    frame0, frame1, initial = parity_scene(device)
    flow, confidence = farneback(
        frame0.to(dtype),
        frame1.to(dtype),
        initial_flow=initial.to(dtype),
        levels=2,
        iterations=3,
        return_confidence=True,
    )
    assert flow.dtype == confidence.dtype == dtype
    assert flow.shape == (1, 2, *PARITY_SIZE) and confidence.shape == (1, *PARITY_SIZE)
    reference = farneback(frame0.double(), frame1.double(), initial_flow=initial.double(), levels=2, iterations=3)
    torch.testing.assert_close(flow.double(), reference, rtol=0, atol=1e-4 if dtype == torch.float32 else 1e-10)


def test_half_precision_runs_in_float32(device: torch.device) -> None:
    frame0, frame1, _ = parity_scene(device)
    frame0, frame1 = frame0.half(), frame1.half()
    flow = farneback(frame0, frame1, levels=2, iterations=3)
    assert flow.dtype == torch.float16
    expected = farneback(frame0.float(), frame1.float(), levels=2, iterations=3)
    torch.testing.assert_close(flow, expected.half(), rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.parametrize("window", ["box", "gaussian"])
def test_cpu_and_cuda_agree(window: str, dtype: torch.dtype) -> None:
    frame0, frame1, initial = parity_scene()
    kwargs = dict(window=window, initial_flow=initial.to(dtype), return_confidence=True)
    flow, confidence = farneback(frame0.to(dtype), frame1.to(dtype), **kwargs)
    kwargs["initial_flow"] = initial.to("cuda", dtype)
    flow_cuda, confidence_cuda = farneback(frame0.to("cuda", dtype), frame1.to("cuda", dtype), **kwargs)
    atol = 1e-4 if dtype == torch.float32 else 1e-10
    torch.testing.assert_close(flow_cuda.cpu(), flow, rtol=0, atol=atol)
    torch.testing.assert_close(confidence_cuda.cpu(), confidence, rtol=atol, atol=atol)


# ----------------------------------------------------------------------------------------------
# Value range
# ----------------------------------------------------------------------------------------------


def range_pair(device: torch.device, dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, ...]:
    """0-255 frames of a rotation and translation, 64x72, and the true flow."""
    true = (rotation_flow(64, 72, 2.0) + translation_flow(64, 72, 1.5, -0.8)).to(device, dtype)
    return (*make_pair(AnalyticTexture(5), true), true)


@pytest.mark.parametrize("scale", [1e-3, 1 / 255, 1e4])
def test_data_range_scales_with_the_frames(scale: float, device: torch.device, dtype: torch.dtype) -> None:
    frame0, frame1, _ = range_pair(device, dtype)
    weights = torch.randn(1, 2, 64, 72, device=device, dtype=dtype)
    kwargs = dict(levels=1, iterations=3, window="gaussian", return_confidence=True)

    def run(f0: torch.Tensor, f1: torch.Tensor, **extra: float) -> tuple[torch.Tensor, ...]:
        f0 = f0.clone().requires_grad_()
        flow, confidence = farneback(f0, f1, **kwargs, **extra)
        return flow, confidence, *torch.autograd.grad((flow * weights).sum(), f0)

    flow, confidence, grad = run(frame0, frame1)
    default = run(frame0, frame1, data_range=255.0)
    assert torch.equal(default[0], flow) and torch.equal(default[1], confidence)
    # frames and data_range scaled alike: the same flow, the confidence times scale ** 2 and the
    # gradients divided by scale. Measured: flows within 1e-5 px in float32 and 2e-14 px in float64,
    # confidences and gradients within 3e-6 and 5e-15 of their largest values.
    tol = 1e-4 if dtype == torch.float32 else 1e-10
    out = run(scale * frame0, scale * frame1, data_range=255 * scale)
    torch.testing.assert_close(out[0], flow, rtol=0, atol=tol)
    torch.testing.assert_close(out[1] / scale**2, confidence, rtol=0, atol=tol * confidence.abs().max().item())
    torch.testing.assert_close(out[2] * scale, grad, rtol=0, atol=tol * grad.abs().max().item())
    # which is eps * (data_range / 255) ** 4 on the frames as given
    rescaled_eps = farneback(scale * frame0, scale * frame1, eps=1e-3 * scale**4, **kwargs)[0]
    torch.testing.assert_close(rescaled_eps, flow, rtol=0, atol=tol)


def test_module_data_range(device: torch.device) -> None:
    # eps is stored in units of 0-255 frames, so a learned value carries over to another data_range
    frame0, frame1, true = range_pair(device)
    kwargs = dict(levels=1, iterations=3, window="gaussian", learnable=True)
    model = Farneback(**kwargs).to(device)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter += 0.5  # as if trained
    unit = Farneback(data_range=1.0, **kwargs).to(device)
    unit.load_state_dict(model.state_dict())
    assert "data_range=255.0" in repr(model) and "data_range=1.0" in repr(unit)
    outs = []
    for m, scale in ((model, 1.0), (unit, 1 / 255)):
        flow = m(scale * frame0, scale * frame1)
        outs.append((flow, torch.autograd.grad((flow - true).norm(dim=1).mean(), list(m.parameters()))))
    torch.testing.assert_close(outs[1][0], outs[0][0], rtol=0, atol=1e-4)
    torch.testing.assert_close(outs[1][1], outs[0][1], rtol=1e-4, atol=0)


def test_frames_in_0_1_need_their_data_range(device: torch.device) -> None:
    # With the default data_range, eps outweighs the systems of frames in [0, 1] and the flow stays
    # near zero, a visible failure, as in OpenCV. Measured: |flow| up to 1.4e-4 px (OpenCV's too) for
    # motions up to 2.6 px; with data_range=1.0, mean error 0.073 px, as on the 0-255 frames.
    frame0, frame1, true = range_pair(device)
    unit0, unit1 = frame0 / 255, frame1 / 255
    limit = 0.001 * true.abs().amax().item()
    assert farneback(unit0, unit1).abs().amax().item() < limit
    assert np.abs(cv_farneback(unit0, unit1, DEFAULTS)).max() < limit
    error = (farneback(unit0, unit1, data_range=1.0) - true).norm(dim=1)[:, 12:-12, 12:-12]
    assert error.mean().item() < 0.1


# ----------------------------------------------------------------------------------------------
# Confidence
# ----------------------------------------------------------------------------------------------


def test_min_eigenvalue() -> None:
    a = torch.randn(4, 3, 6, 7, dtype=torch.float64)
    b = torch.randn(4, 3, 6, 7, dtype=torch.float64)
    g11, g12, g22 = a.square() + b.square(), a * b, b.square() + 2 * a.square()
    averaged = torch.stack((g11, g12, g22, a, b), dim=1).flatten(2, 3)
    matrices = torch.stack((torch.stack((g11, g12), -1), torch.stack((g12, g22), -1)), -2)
    expected = torch.linalg.eigvalsh(matrices)[..., 0]
    torch.testing.assert_close(fb._min_eigenvalue(averaged), expected.flatten(1, 2))


def test_confidence_is_the_smallest_eigenvalue_of_the_averaged_system() -> None:
    frame0, frame1, _ = parity_scene()
    frame0, frame1 = frame0.double(), frame1.double()
    kwargs = dict(levels=0, winsize=9, window="gaussian")
    first = farneback(frame0, frame1, iterations=1, **kwargs)
    _, confidence = farneback(frame0, frame1, iterations=2, return_confidence=True, **kwargs)
    # the system of the last iteration, built at the flow of the first one
    images = pyramid_level(torch.cat((frame0, frame1)), pyramid_levels(*PARITY_SIZE, 0.5, 0)[0])
    r = fb._poly_exp(images, fb._poly_kernels(5, 1.1, frame0))
    matrices = fb._update_matrices(r[:1, None], r[1:, None], first, fb._border_scale(*PARITY_SIZE, frame0))
    kernel = fb._window_kernel("gaussian", 9, None, frame0)
    g11, g12, g22, _, _ = separable_filter(matrices, kernel, kernel, border="replicate").unbind(1)
    systems = torch.stack((g11, g12, g12, g22), -1).unflatten(-1, (2, 2))
    torch.testing.assert_close(confidence, torch.linalg.eigvalsh(systems)[..., 0])


def test_confidence_reflects_the_aperture_problem() -> None:
    height, width = 64, 80
    texture = smooth_texture(height, width, seed=5)
    stripes = 127.5 + 100 * torch.sin(torch.arange(width) / 3.0).expand(1, 1, height, width)
    _, textured = farneback(texture, torch.roll(texture, 1, -1), return_confidence=True)
    _, striped = farneback(stripes, torch.roll(stripes, 1, -1), return_confidence=True)
    assert (textured >= 0).all() and (striped >= -1e-3).all()
    assert striped[:, 16:-16, 16:-16].max() < 1e-3 * textured[:, 16:-16, 16:-16].median()


# ----------------------------------------------------------------------------------------------
# torch.compile
# ----------------------------------------------------------------------------------------------


@pytest.mark.compile
@pytest.mark.slow
def test_compile(device: torch.device) -> None:
    frame0, frame1, initial = parity_scene(device)
    frame0, frame1, initial = frame0[..., :67, :81], frame1[..., :67, :81], initial[..., :67, :81]
    sigma = torch.tensor(1.3, device=device, requires_grad=True)

    def fn(sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        kwargs = dict(pyr_scale=0.6, levels=1, iterations=2, winsize=9, window="gaussian", poly_sigma=sigma)
        return farneback(frame0, frame1, initial_flow=initial, return_confidence=True, **kwargs)

    expected = fn(sigma)
    (grad,) = torch.autograd.grad(expected[0].square().sum() + expected[1].sum(), sigma)
    out = torch.compile(fn, fullgraph=True)(sigma)
    (compiled_grad,) = torch.autograd.grad(out[0].square().sum() + out[1].sum(), sigma)
    # Inductor reorders float32 arithmetic, a few 1e-5 relative after the iterations
    torch.testing.assert_close(out[0], expected[0], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out[1], expected[1], rtol=1e-5, atol=1e-3)
    torch.testing.assert_close(compiled_grad, grad, rtol=1e-4, atol=0)


# ----------------------------------------------------------------------------------------------
# Module
# ----------------------------------------------------------------------------------------------


def test_module_matches_function() -> None:
    frame0, frame1, initial = parity_scene()
    kwargs = dict(levels=2, iterations=3, window="gaussian", window_sigma=2.0, poly_sigma=1.3, eps=0.01)
    model = Farneback(**kwargs)
    assert list(model.parameters()) == []
    expected = farneback(frame0, frame1, initial_flow=initial, return_confidence=True, **kwargs)
    out = model(frame0, frame1, initial, return_confidence=True)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    learned = Farneback(learnable=True, **kwargs)
    torch.testing.assert_close(learned(frame0, frame1, initial), expected[0], rtol=0, atol=1e-4)


def test_module_learnable_parameters() -> None:
    names = {name for name, _ in Farneback(learnable=True).named_parameters()}
    assert names == {"log_poly_sigma", "log_eps"}
    model = Farneback(learnable=True, window="gaussian", winsize=15, poly_sigma=1.5, eps=0.1)
    assert {name for name, _ in model.named_parameters()} == {"log_poly_sigma", "log_window_sigma", "log_eps"}
    torch.testing.assert_close(model.window_sigma, torch.tensor(0.3 * 7))
    torch.testing.assert_close(model.poly_sigma, torch.tensor(1.5))
    torch.testing.assert_close(model.eps, torch.tensor(0.1))
    # stored as logarithms, so that any step keeps the values positive
    for name, value in (("poly_sigma", 1.5), ("window_sigma", 0.3 * 7), ("eps", 0.1)):
        torch.testing.assert_close(getattr(model, f"log_{name}").detach(), torch.tensor(math.log(value)))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter -= 50
    assert model.poly_sigma > 0 and model.window_sigma > 0 and model.eps > 0
    model = Farneback(learnable=["eps"], window="gaussian")
    assert [name for name, _ in model.named_parameters()] == ["log_eps"]
    assert model.poly_sigma == 1.1 and model.window_sigma is None
    assert [name for name, _ in Farneback(learnable="poly_sigma").named_parameters()] == ["log_poly_sigma"]
    assert "learnable=['eps']" in repr(Farneback(learnable={"eps"}))


def test_module_training_reduces_the_error() -> None:
    height, width = 64, 72
    true = rotation_flow(height, width, 2.0) + translation_flow(height, width, 0.6, -0.4)
    frame0, frame1 = make_pair(AnalyticTexture(5), true)
    frame0 = frame0 + 4 * torch.randn_like(frame0)  # noise gives eps and the sigmas something to do
    frame1 = frame1 + 4 * torch.randn_like(frame1)
    model = Farneback(
        levels=0,
        iterations=3,
        winsize=15,
        window="gaussian",
        poly_sigma=0.6,
        window_sigma=0.8,
        eps=10.0,
        learnable=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    losses = []
    for _ in range(20):
        optimizer.zero_grad()
        loss = (model(frame0, frame1) - true).norm(dim=1)[:, 8:-8, 8:-8].mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < 0.6 * losses[0]


@pytest.mark.cuda
def test_mixed_precision_training_step(grad_scaler: torch.amp.GradScaler) -> None:
    # float32 frames inside autocast(float16), with a GradScaler. Under autocast, the float16
    # convolutions used to give non-finite gradients, and the step was skipped.
    true = rotation_flow(96, 112, 2.0, device="cuda") + translation_flow(96, 112, 1.5, -0.8, device="cuda")
    frame0, frame1 = (frame.requires_grad_() for frame in make_pair(AnalyticTexture(5), true))
    model = Farneback(window="gaussian", learnable=True).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    with torch.autocast("cuda", dtype=torch.float16):
        flow = model(frame0, frame1)
        loss = (flow - true).norm(dim=1).mean()
    grad_scaler.scale(loss).backward()
    grad_scaler.step(optimizer)
    grad_scaler.update()
    assert flow.dtype == torch.float32 and flow.isfinite().all()
    assert all(tensor.grad.isfinite().all() for tensor in (frame0, frame1, *model.parameters()))
    assert grad_scaler.get_scale() == 2.0**16  # halved if the step had been skipped for non-finite gradients


# ----------------------------------------------------------------------------------------------
# Argument errors
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        (dict(pyr_scale=1.0), ValueError, "pyr_scale"),
        (dict(pyr_scale=0.0), ValueError, "pyr_scale"),
        (dict(pyr_scale="0.5"), TypeError, "pyr_scale"),
        (dict(levels=-1), ValueError, "levels"),
        (dict(levels=2.0), TypeError, "levels"),
        (dict(levels=True), TypeError, "levels"),
        (dict(winsize=0), ValueError, "winsize"),
        (dict(iterations=0), ValueError, "iterations"),
        (dict(poly_n=0), ValueError, "poly_n"),
        (dict(poly_sigma="1"), TypeError, "poly_sigma"),
        (dict(poly_sigma=float("nan")), ValueError, "poly_sigma"),
        (dict(poly_sigma=torch.ones(2)), ValueError, "poly_sigma"),
        (dict(poly_sigma=torch.tensor(1)), TypeError, "poly_sigma"),
        (dict(window="median"), ValueError, "window"),
        (dict(window_sigma=2.0), ValueError, "window_sigma is only valid"),
        (dict(window="gaussian", window_sigma=0.0), ValueError, "window_sigma"),
        (dict(eps=-1.0), ValueError, "eps"),
        (dict(eps=0.0), ValueError, "eps must be positive"),
        (dict(eps=None), TypeError, "eps"),
        (dict(data_range=0.0), ValueError, "data_range must be positive"),
        (dict(data_range=-255), ValueError, "data_range must be positive"),
        (dict(data_range=float("inf")), ValueError, "data_range must be finite"),
        (dict(data_range=float("nan")), ValueError, "data_range must be finite"),
        (dict(data_range="255"), TypeError, "data_range must be a float"),
        (dict(data_range=None), TypeError, "data_range must be a float"),
        (dict(data_range=True), TypeError, "data_range must be a float"),
        (dict(data_range=torch.tensor(255.0)), TypeError, "data_range must be a float"),
    ],
)
def test_argument_errors(kwargs: dict, error: type[Exception], match: str) -> None:
    frame = torch.zeros(1, 1, 8, 8)
    with pytest.raises(error, match=match):
        farneback(frame, frame, **kwargs)
    if not any(isinstance(v, torch.Tensor) for v in kwargs.values()):  # the module takes floats only
        with pytest.raises(error, match=match):
            Farneback(**kwargs)


def test_frame_and_flow_errors() -> None:
    frame = torch.zeros(1, 1, 8, 8)
    with pytest.raises(ValueError, match="same shape"):
        farneback(frame, torch.zeros(1, 1, 8, 9))
    with pytest.raises(TypeError, match="floating-point"):
        farneback(frame.long(), frame.long())
    with pytest.raises(ValueError, match="initial_flow"):
        farneback(frame, frame, initial_flow=torch.zeros(1, 2, 8, 9))
    with pytest.raises(TypeError, match="initial_flow"):
        farneback(frame, frame, initial_flow=torch.zeros(1, 2, 8, 8, dtype=torch.float64))


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        (dict(learnable=["sigma"]), ValueError, "learnable parameters must be among"),
        (dict(learnable=["window_sigma"]), ValueError, "window='gaussian'"),
        (dict(learnable=["window_sigma"], window="gaussian", winsize=1), ValueError, "winsize >= 2"),
        (dict(learnable=3), TypeError, "learnable"),
        (dict(learnable=True, poly_sigma=0.0), ValueError, "poly_sigma must be positive"),
        (dict(learnable=["eps"], eps=0.0), ValueError, "eps must be positive"),
        (dict(poly_sigma=torch.tensor(1.0)), TypeError, "poly_sigma must be a float"),
        (dict(data_range=torch.tensor(1.0)), TypeError, "data_range must be a float"),
    ],
)
def test_module_errors(kwargs: dict, error: type[Exception], match: str) -> None:
    with pytest.raises(error, match=match):
        Farneback(**kwargs)

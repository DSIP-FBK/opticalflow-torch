"""Tests of lucas_kanade and LucasKanade."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from synthetic import (
    AnalyticTexture,
    image_to_cv,
    make_pair,
    rotation_flow,
    smooth_texture,
    translation_flow,
    zoom_flow,
)
from torch.autograd import gradcheck

from opticalflow_torch import LucasKanade, lucas_kanade, warp
from opticalflow_torch.lucas_kanade import _window_kernel

cv2 = pytest.importorskip("cv2")

MOTIONS = {
    "subpixel": lambda h, w: translation_flow(h, w, 0.37, -0.62, dtype=torch.float64),
    "rotation": lambda h, w: rotation_flow(h, w, 3.0, dtype=torch.float64),
    "zoom": lambda h, w: zoom_flow(h, w, 1.04, dtype=torch.float64),
}


def interior(flow: torch.Tensor, margin: int) -> torch.Tensor:
    """Pixels at least ``margin`` pixels from the borders, before and after displacement, shape (B, H, W)."""
    height, width = flow.shape[-2:]
    x = torch.arange(width, device=flow.device) + flow[:, 0]
    y = torch.arange(height, device=flow.device)[:, None] + flow[:, 1]
    mask = (x >= margin) & (x <= width - 1 - margin) & (y >= margin) & (y <= height - 1 - margin)
    mask[:, :margin] = mask[:, -margin:] = False
    mask[:, :, :margin] = mask[:, :, -margin:] = False
    return mask


def endpoint_error(estimate: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    return (estimate - flow).norm(dim=1)


# ----------------------------------------------------------------------------------------------
# Accuracy
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("window", ["gaussian", "box"])
@pytest.mark.parametrize("name", list(MOTIONS))
def test_synthetic_motions(device: torch.device, dtype: torch.dtype, name: str, window: str) -> None:
    flow = MOTIONS[name](96, 112).to(device, dtype)
    frame0, frame1 = make_pair(AnalyticTexture(1), flow)
    error = endpoint_error(lucas_kanade(frame0, frame1, window=window), flow)
    inner = error[interior(flow, 10)]
    # Measured on seeds 1-3, identical on both devices and dtypes: inner mean 0.018-0.038 (Gaussian
    # window) and 0.009-0.020 (box, effectively larger), inner max 0.08-0.17, mean over all pixels
    # up to 0.07 (pixels near the borders see part of their window outside frame1). The flows
    # reach 0.7 (translation), 3.9 (rotation) and 2.9 (zoom) pixels.
    assert inner.mean().item() < 0.05 and inner.amax().item() < 0.25
    assert error.mean().item() < 0.1


def test_large_translation_needs_the_pyramid(device: torch.device) -> None:
    # 7.8 px, beyond the reach of a 21 px window at full resolution with textures of 8-64 px waves
    flow = translation_flow(128, 144, 6.3, -4.6, device=device)
    frame0, frame1 = make_pair(AnalyticTexture(2), flow)
    mask = interior(flow, 16)
    # measured (seeds 2-4): 4.0-8.3 px without levels; 0.028-0.030 px (max 0.14) with two, and the
    # same with the initial flow below
    assert endpoint_error(lucas_kanade(frame0, frame1, levels=0), flow)[mask].mean().item() > 1
    error = endpoint_error(lucas_kanade(frame0, frame1, levels=2), flow)[mask]
    assert error.mean().item() < 0.05 and error.amax().item() < 0.3
    # an initial flow close to the truth makes the pyramid unnecessary, and is resized for it
    initial = translation_flow(128, 144, 6.0, -4.0, device=device)
    for levels in (0, 2):
        error = endpoint_error(lucas_kanade(frame0, frame1, levels=levels, initial_flow=initial), flow)[mask]
        assert error.mean().item() < 0.05 and error.amax().item() < 0.3
    # other size ratios; measured mean 0.028-0.031, max 0.14
    for pyr_scale, levels in ((0.7, 3), (0.6, 2)):
        error = endpoint_error(lucas_kanade(frame0, frame1, pyr_scale=pyr_scale, levels=levels), flow)[mask]
        assert error.mean().item() < 0.05 and error.amax().item() < 0.3


def test_large_translation_default_settings(device: torch.device) -> None:
    # 20 px, 2.5 px on the coarsest level (32x36). Without the extra blur of the coarser levels, 5
    # iterations left 12-22 px on these seeds, worse than a zero flow.
    frame1 = torch.cat([smooth_texture(256, 288, sigma=1.5, seed=seed, device=device) for seed in range(3)])
    frame0 = torch.roll(frame1, (12, -16), (-2, -1))  # periodic texture: frame0(p) = frame1(p + (16, -12))
    flow = translation_flow(256, 288, 16.0, -12.0, batch=3, device=device)
    for window in ("gaussian", "box"):
        error = endpoint_error(lucas_kanade(frame0, frame1, window=window), flow)[:, 32:-32, 32:-32]
        # measured: mean 6e-5 (Gaussian window) and 8e-4 (box), max 0.006 and 0.038
        assert error.mean().item() < 0.005 and error.amax().item() < 0.1


def test_pyramid_skips_levels_under_32_pixels(device: torch.device) -> None:
    flow = MOTIONS["zoom"](64, 72).to(device)
    frame0, frame1 = make_pair(AnalyticTexture(1), flow)
    no_pyramid = lucas_kanade(frame0, frame1, levels=0)
    # at 64x72 the first coarser level would be 26x29 (skipped) with pyr_scale 0.4, and is 32x36 with
    # 0.5 and 45x50 with 0.7; the second one is under 32 pixels with all three
    assert torch.equal(lucas_kanade(frame0, frame1, pyr_scale=0.4, levels=3), no_pyramid)
    for pyr_scale in (0.5, 0.7):
        one = lucas_kanade(frame0, frame1, pyr_scale=pyr_scale, levels=1)
        assert torch.equal(lucas_kanade(frame0, frame1, pyr_scale=pyr_scale, levels=3), one)
        assert (one - no_pyramid).abs().amax().item() > 0.02  # measured 0.057 and 0.067


def test_channels_sum_their_equations(device: torch.device) -> None:
    # each channel alone has the aperture problem: stripes along y in channel 0, along x in channel 1
    height, width, dx, dy = 64, 72, 0.6, -0.4

    def stripes(shift_x: float, shift_y: float) -> torch.Tensor:
        x = torch.arange(width, dtype=torch.float64) + shift_x
        y = torch.arange(height, dtype=torch.float64)[:, None] + shift_y
        channels = (
            (127.5 + 100 * torch.cos(2 * math.pi * x / 13 + 0.3)).expand(height, width),
            (127.5 + 100 * torch.cos(2 * math.pi * y / 11 + 1.1)).expand(height, width),
        )
        return torch.stack(channels).unsqueeze(0).to(device)

    frame0, frame1 = stripes(dx, dy), stripes(0, 0)
    flow = translation_flow(height, width, dx, dy, device=device, dtype=torch.float64)
    mask = interior(flow, 10)
    # measured: max 0.0044 px, and 0.0033 px for x from channel 0 alone
    assert endpoint_error(lucas_kanade(frame0, frame1), flow)[mask].amax().item() < 0.02
    # with channel 0 only, x is found and y stays exactly at its start (no y-gradients at all)
    out = lucas_kanade(frame0[:, :1], frame1[:, :1])
    assert (out[:, 0] - dx)[mask].abs().amax().item() < 0.02
    assert (out[:, 1] == 0).all()


def test_even_box_window(device: torch.device) -> None:
    torch.testing.assert_close(_window_kernel("box", 4, None, torch.zeros(1)), torch.tensor([1, 2, 2, 2, 1]) / 8)
    torch.testing.assert_close(_window_kernel("box", 5, None, torch.zeros(1)), torch.full((5,), 0.2))
    flow = MOTIONS["rotation"](96, 112).to(device)
    frame0, frame1 = make_pair(AnalyticTexture(1), flow)
    error = endpoint_error(lucas_kanade(frame0, frame1, window="box", winsize=20), flow)[interior(flow, 10)]
    assert error.mean().item() < 0.05  # measured 0.018


def test_eps_scales_with_squared_intensities(device: torch.device) -> None:
    # a smooth texture at a quarter of the contrast, whose gradients are small enough for eps to matter
    flow = translation_flow(160, 176, 3.0, -2.0, device=device, dtype=torch.float64)
    texture = AnalyticTexture(1, min_wavelength=32, max_wavelength=256)
    frame0, frame1 = (127.5 + 0.25 * (frame - 127.5) for frame in make_pair(texture, flow))
    reference = lucas_kanade(frame0, frame1)
    for scale in (1 / 255, 10.0):
        for kwargs in ({"eps": 1e-3 * scale**2}, {"data_range": 255 * scale}):
            scaled = lucas_kanade(scale * frame0, scale * frame1, **kwargs)
            torch.testing.assert_close(scaled, reference, rtol=0, atol=1e-9)  # measured 1.3e-13
    # the default data_range is 255: measured 0.020 px, and 1.55 px on the same frames in [0, 1]
    mask = interior(flow, 16)
    assert endpoint_error(reference, flow)[mask].mean().item() < 0.05
    assert endpoint_error(lucas_kanade(frame0 / 255, frame1 / 255), flow)[mask].mean().item() > 0.5


@pytest.mark.parametrize("scale", [1e-3, 1 / 255, 1e4])
def test_data_range_scales_with_the_frames(scale: float, device: torch.device, dtype: torch.dtype) -> None:
    flow = rotation_flow(64, 72, 3.0, device=device, dtype=dtype) + 1.5
    frame0, frame1 = make_pair(AnalyticTexture(5), flow)
    weights = torch.randn(1, 2, 64, 72, device=device, dtype=dtype)
    options = {"levels": 1, "iterations": 3, "return_confidence": True}

    def run(f0: torch.Tensor, f1: torch.Tensor, **extra: float) -> tuple[torch.Tensor, ...]:
        f0 = f0.clone().requires_grad_()
        out, confidence = lucas_kanade(f0, f1, **options, **extra)
        return out, confidence, *torch.autograd.grad((out * weights).sum(), f0)

    out, confidence, grad = run(frame0, frame1)
    default = run(frame0, frame1, data_range=255.0)
    assert torch.equal(default[0], out) and torch.equal(default[1], confidence)
    # frames and data_range scaled alike: the same flow, the confidence times scale ** 2 and the
    # gradients divided by scale. Measured: flows within 3e-6 px in float32 and 4e-15 px in float64,
    # confidences and gradients within 3e-6 and 5e-15 of their largest values.
    tol = 1e-4 if dtype == torch.float32 else 1e-10
    scaled = run(scale * frame0, scale * frame1, data_range=255 * scale)
    torch.testing.assert_close(scaled[0], out, rtol=0, atol=tol)
    torch.testing.assert_close(scaled[1] / scale**2, confidence, rtol=0, atol=tol * confidence.abs().max().item())
    torch.testing.assert_close(scaled[2] * scale, grad, rtol=0, atol=tol * grad.abs().max().item())
    # which is eps * (data_range / 255) ** 2 on the frames as given
    rescaled_eps = lucas_kanade(scale * frame0, scale * frame1, eps=1e-3 * scale**2, **options)[0]
    torch.testing.assert_close(rescaled_eps, out, rtol=0, atol=tol)


def test_residual_damping_spares_explained_motions(device: torch.device) -> None:
    # Smooth textures give nearly rank-one structure tensors. Damping by the whole squared residual,
    # rather than by the part that no displacement explains, held back the weak direction: 0.11 and
    # 0.30 px here, against 0.027 and 0.039 px (0.027 and 0.034 px without any residual damping).
    texture = AnalyticTexture(1, min_wavelength=32, max_wavelength=256)
    for dx, dy in ((1.0, -0.7), (2.0, -1.5)):
        flow = translation_flow(160, 176, dx, dy, device=device, dtype=torch.float64)
        frame0, frame1 = make_pair(texture, flow)
        error = endpoint_error(lucas_kanade(frame0, frame1, levels=0), flow)[interior(flow, 16)]
        assert error.mean().item() < 0.06


# ----------------------------------------------------------------------------------------------
# OpenCV
# ----------------------------------------------------------------------------------------------


def opencv_sparse_lk(frame0: torch.Tensor, frame1: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """``goodFeaturesToTrack`` corners of ``frame0`` (x, y) and their flow by ``calcOpticalFlowPyrLK``."""
    image0, image1 = image_to_cv(frame0).astype(np.uint8), image_to_cv(frame1).astype(np.uint8)
    points = cv2.goodFeaturesToTrack(image0, maxCorners=300, qualityLevel=0.01, minDistance=8, blockSize=7)
    height, width = image0.shape
    inside = (points[:, 0, 0] >= 24) & (points[:, 0, 0] < width - 24) & (points[:, 0, 1] >= 24)
    points = points[inside & (points[:, 0, 1] < height - 24)]
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 0.01)
    moved, status, _ = cv2.calcOpticalFlowPyrLK(
        image0, image1, points, None, winSize=(21, 21), maxLevel=3, criteria=criteria
    )
    found = status[:, 0] == 1
    return points[found, 0], (moved - points)[found, 0]


OPENCV_MOTIONS = {
    "subpixel": lambda h, w: translation_flow(h, w, 0.37, -0.62, dtype=torch.float64),
    "translation": lambda h, w: translation_flow(h, w, 6.3, -4.6, dtype=torch.float64),
    "rotation": lambda h, w: rotation_flow(h, w, 2.0, dtype=torch.float64),
}


@pytest.mark.parametrize("name", list(OPENCV_MOTIONS))
def test_opencv_sparse_lk(name: str) -> None:
    # OpenCV's sparse LK on 8-bit frames with the same window (21 x 21 box) and pyramid depth
    flow = OPENCV_MOTIONS[name](256, 288)
    frame0, frame1 = (frame.round().clamp(0, 255) for frame in make_pair(AnalyticTexture(0), flow))
    points, cv_flow = opencv_sparse_lk(frame0, frame1)
    assert len(points) > 150
    out = lucas_kanade(frame0.float(), frame1.float(), window="box", winsize=21, levels=3)
    cols, rows = np.rint(points[:, 0]).astype(int), np.rint(points[:, 1]).astype(int)  # corners are integer
    ours = out[0].numpy()[:, rows, cols].T
    truth = flow[0].numpy()[:, rows, cols].T
    ours_error, cv_error = np.linalg.norm(ours - truth, axis=1), np.linalg.norm(cv_flow - truth, axis=1)
    if name == "rotation":
        # OpenCV fits a constant flow to each window, which misses the 0.37 px variation of a 2 degree
        # rotation across 21 px; the flow smoothing keeps it. Measured (seeds 0-2): OpenCV mean 0.078,
        # max 0.27; ours mean 0.007, max 0.033.
        assert ours_error.mean() < 0.02 and ours_error.max() < 0.08
        assert ours_error.mean() < 0.3 * cv_error.mean()
        return
    # Measured (seeds 0-2): mean difference 0.0021-0.0023, max 0.008-0.016; both are within 0.008
    # of the truth on average (the frames are rounded to integers).
    difference = np.linalg.norm(ours - cv_flow, axis=1)
    assert difference.mean() < 0.006 and difference.max() < 0.04
    assert ours_error.mean() < 0.02 and cv_error.mean() < 0.02


@pytest.mark.parametrize("window", ["gaussian", "box"])
def test_confidence_matches_opencv_structure_tensor(window: str) -> None:
    # with frame1 = frame0 the flow stays zero (up to grid_sample's rounding of positions), so the
    # confidence is the smallest eigenvalue of the window-averaged structure tensor of the level-0
    # frame (blurred by [1, 2, 1] / 4)
    frame = AnalyticTexture(3, channels=2).image(40, 52, dtype=torch.float64)
    flow, confidence = lucas_kanade(frame, frame, window=window, winsize=9, return_confidence=True)
    assert flow.abs().amax().item() < 1e-12
    image = cv2.GaussianBlur(frame[0].permute(1, 2, 0).numpy(), (3, 3), 0)
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=1, borderType=cv2.BORDER_REPLICATE) / 2
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=1, borderType=cv2.BORDER_REPLICATE) / 2

    def average(values: np.ndarray) -> np.ndarray:
        if window == "box":
            return cv2.blur(values.sum(-1), (9, 9), borderType=cv2.BORDER_REFLECT_101)
        return cv2.GaussianBlur(values.sum(-1), (9, 9), 0.3 * 4, borderType=cv2.BORDER_REFLECT_101)

    g11, g12, g22 = average(gx * gx), average(gx * gy), average(gy * gy)
    expected = np.linalg.eigvalsh(np.stack((np.stack((g11, g12), -1), np.stack((g12, g22), -1)), -2))[..., 0]
    torch.testing.assert_close(confidence[0], torch.from_numpy(expected), rtol=0, atol=1e-9 * expected.max())


def test_confidence_drops_outside_frame1(device: torch.device) -> None:
    flow = translation_flow(96, 112, 8.0, 0.0, device=device)
    frame0, frame1 = make_pair(AnalyticTexture(1), flow)
    _, confidence = lucas_kanade(frame0, frame1, return_confidence=True)
    assert confidence.shape == (1, 96, 112) and confidence.dtype == frame0.dtype
    # the last 8 columns are displaced outside frame1; measured ratio of medians 0.011
    assert confidence[..., -4:].median() < 0.05 * confidence[..., 20:-20].median()


# ----------------------------------------------------------------------------------------------
# Degenerate inputs and dtypes
# ----------------------------------------------------------------------------------------------


def test_constant_frames(device: torch.device, dtype: torch.dtype) -> None:
    frame0 = torch.full((2, 3, 64, 72), 100.0, device=device, dtype=dtype, requires_grad=True)
    frame1 = torch.full((2, 3, 64, 72), 130.0, device=device, dtype=dtype, requires_grad=True)
    sigma = torch.tensor(2.0, device=device, dtype=dtype, requires_grad=True)
    eps = torch.tensor(1e-3, device=device, dtype=dtype, requires_grad=True)
    flow, confidence = lucas_kanade(frame0, frame1, levels=1, window_sigma=sigma, eps=eps, return_confidence=True)
    assert (flow == 0).all() and (confidence == 0).all()
    (flow.sum() + confidence.sum()).backward()
    for tensor in (frame0, frame1, sigma, eps):
        assert tensor.grad.isfinite().all()


def test_flat_frame0_with_brighter_frame1(device: torch.device) -> None:
    # No displacement explains the change. With eps alone, the update reached r / (2 sqrt(eps)) where
    # the texture of frame0 is about sqrt(eps), and its derivative r / eps where frame0 is flat.
    frame0 = torch.full((1, 1, 64, 64), 100.0, device=device, dtype=torch.float64, requires_grad=True)
    frame1 = torch.full_like(frame0, 130.0)
    (lucas_kanade(frame0, frame1) * torch.randn(1, 2, 64, 64, device=device, dtype=torch.float64)).sum().backward()
    assert frame0.grad.abs().amax().item() < 5  # measured 0.6-1.1; was 4e4-7e4
    # faint texture in both frames; measured up to 0.14 px, was 7-150 px
    x = torch.arange(64, device=device, dtype=torch.float64)
    texture = (torch.sin(2 * math.pi * x / 9.3 + 0.4) * torch.cos(2 * math.pi * x[:, None] / 7.1)).expand_as(frame0)
    for amplitude in (1e-3, 1e-2, 0.1, 1.0):
        assert lucas_kanade(100 + amplitude * texture, 130 + amplitude * texture).abs().amax().item() < 0.5
    noisy = frame0.detach() + 0.01 * torch.randn_like(frame0)
    assert lucas_kanade(noisy, frame1).abs().amax().item() < 0.05  # measured 0.004; was 78-91 px


def test_rain_appearing_in_a_dry_area(device: torch.device) -> None:
    # radar-like, in 0-255: a cell moves by (1, 0.5) px and a new one appears where frame0 is exactly 0
    x = torch.arange(96, device=device, dtype=torch.float64)
    y = x[:, None]

    def cell(cx: float, cy: float) -> torch.Tensor:
        value = 40 * torch.exp(-((x - cx) ** 2 + (y - cy) ** 2) / 72) + 15 * torch.exp(
            -((x - cx + 2) ** 2 + (y - cy - 3) ** 2) / 18
        )
        return torch.where(value < 0.5, 0.0, value).expand(1, 1, 96, 96)

    frame0 = (255 / 55 * cell(30, 30)).requires_grad_()
    frame1 = 255 / 55 * (cell(31, 30.5) + cell(65, 65))
    flow = lucas_kanade(frame0, frame1)
    (warp(frame1, flow) - frame0).square().mean().backward()
    dry = frame0.detach() == 0
    # measured: 0.32 on dry pixels and 0.011 on wet ones (farneback: 0.72 and 3e-4); was 1.7e5 and 0.41
    assert frame0.grad[dry].abs().amax().item() < 2 and frame0.grad[~dry].abs().amax().item() < 0.1
    # measured 0.013-0.025 px; was 40-80 px
    noisy = frame0.detach() + 0.01 * torch.randn_like(frame0) * dry
    assert (lucas_kanade(noisy, frame1) - flow).abs().amax().item() < 0.2


@pytest.mark.parametrize("eps", [1e-18, 1e-24, 1e-40])
def test_tiny_eps_with_flat_windows(device: torch.device, eps: float) -> None:
    # float32 frames that are exactly 0 on the right: flat windows used to divide by eps ** 2, which
    # made every gradient NaN from eps = 1e-18 (with this loss scale) and the flow NaN from 1e-23
    frame1 = smooth_texture(64, 64, seed=3, device=device)
    frame1[..., 40:] = 0
    frame0 = torch.roll(frame1, 1, -1)
    tensors = (frame0.requires_grad_(), frame1.requires_grad_(), torch.tensor(eps, device=device, requires_grad=True))
    flow = lucas_kanade(*tensors[:2], eps=tensors[2])
    (1e6 * torch.randn_like(flow) * flow).sum().backward()
    assert flow.isfinite().all()
    assert all(tensor.grad.isfinite().all() for tensor in tensors)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["float16", "bfloat16"])
def test_low_precision_gradients_are_finite(device: torch.device, dtype: torch.dtype) -> None:
    # gradients of a zero frame0 used to reach 2e5, inf in float16
    frame1 = AnalyticTexture(1).image(64, 64, device=device, dtype=dtype)
    frame0 = torch.zeros_like(frame1, requires_grad=True)
    flow = lucas_kanade(frame0, frame1)
    (flow.float() * torch.randn(flow.shape, device=device)).sum().backward()
    assert frame0.grad.isfinite().all() and frame0.grad.abs().amax().item() < 10  # measured 0.2-0.23


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_is_processed_in_float32(device: torch.device, dtype: torch.dtype) -> None:
    flow = MOTIONS["subpixel"](64, 72).to(device)
    frame0, frame1 = (frame.to(dtype) for frame in make_pair(AnalyticTexture(1), flow, dtype=torch.float32))
    out, confidence = lucas_kanade(frame0, frame1, return_confidence=True)
    assert out.dtype == confidence.dtype == dtype
    ref, ref_confidence = lucas_kanade(frame0.float(), frame1.float(), return_confidence=True)
    torch.testing.assert_close(out, ref.to(dtype), rtol=0, atol=0)
    torch.testing.assert_close(confidence, ref_confidence.to(dtype), rtol=0, atol=0)


# ----------------------------------------------------------------------------------------------
# Gradients, batches, devices, compilation
# ----------------------------------------------------------------------------------------------


def gradcheck_inputs(device: torch.device, shape: tuple[int, int, int, int]) -> tuple[torch.Tensor, ...]:
    """Frames in [0, 1] with a rotation, and a non-integer initial flow, all float64 with gradients."""
    batch, channels, height, width = shape
    flow = rotation_flow(height, width, 6.0, batch=batch, dtype=torch.float64)
    texture = AnalyticTexture(4, channels=channels, min_wavelength=6)
    frame0, frame1 = (frame / 255 for frame in make_pair(texture, flow, device=device))
    initial = 0.3 * torch.randn(batch, 2, height, width, device=device, dtype=torch.float64) + 0.05
    return tuple(t.requires_grad_() for t in (frame0, frame1, initial))


def test_gradcheck(device: torch.device) -> None:
    # frames in [0, 1] give structure tensors of about 1e-3, so eps = 2e-3 matters
    frame0, frame1, initial = gradcheck_inputs(device, (1, 2, 10, 12))
    sigma = torch.tensor(1.3, device=device, dtype=torch.float64, requires_grad=True)
    eps = torch.tensor(2e-3, device=device, dtype=torch.float64, requires_grad=True)

    def fn(*args: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f0, f1, init, s, e = args
        options = {"levels": 0, "winsize": 5, "iterations": 3, "return_confidence": True}
        return lucas_kanade(f0, f1, window_sigma=s, eps=e, initial_flow=init, **options)

    # grid_sample's and index_select's backward accumulate with atomics on CUDA
    assert gradcheck(fn, (frame0, frame1, initial, sigma, eps), nondet_tol=1e-12)


def test_gradcheck_pyramid(device: torch.device) -> None:
    frame0, frame1, initial = gradcheck_inputs(device, (2, 1, 64, 66))
    eps = torch.tensor(2e-3, device=device, dtype=torch.float64, requires_grad=True)

    def fn(f0: torch.Tensor, f1: torch.Tensor, init: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        return lucas_kanade(f0, f1, levels=1, winsize=7, iterations=2, window="box", eps=e, initial_flow=init)

    assert gradcheck(fn, (frame0, frame1, initial, eps), fast_mode=True, nondet_tol=1e-12)


def test_batch_matches_single(device: torch.device, dtype: torch.dtype) -> None:
    flow = torch.cat([MOTIONS[name](64, 72) for name in MOTIONS]).to(device, dtype)
    frame0, frame1 = make_pair(AnalyticTexture(5, channels=2), flow)
    initial = 0.5 * flow
    out, confidence = lucas_kanade(frame0, frame1, initial_flow=initial, return_confidence=True)
    for i in range(3):
        single, single_confidence = lucas_kanade(
            frame0[i : i + 1], frame1[i : i + 1], initial_flow=initial[i : i + 1], return_confidence=True
        )
        # measured: identical
        torch.testing.assert_close(out[i : i + 1], single, rtol=0, atol=1e-5)
        torch.testing.assert_close(confidence[i : i + 1], single_confidence, rtol=1e-6, atol=0)


@pytest.mark.cuda
@pytest.mark.parametrize("window", ["gaussian", "box"])
def test_cpu_cuda_agree(dtype: torch.dtype, window: str) -> None:
    flow = rotation_flow(80, 96, 3.0, batch=2, dtype=dtype)
    frame0, frame1 = make_pair(AnalyticTexture(0, channels=2), flow)
    out, confidence = lucas_kanade(frame0, frame1, window=window, return_confidence=True)
    out_cuda, confidence_cuda = lucas_kanade(frame0.cuda(), frame1.cuda(), window=window, return_confidence=True)
    # measured: flows within 7e-6 px (float32) and 1.3e-14 px (float64)
    atol = 1e-4 if dtype == torch.float32 else 1e-11
    torch.testing.assert_close(out_cuda.cpu(), out, rtol=0, atol=atol)
    torch.testing.assert_close(confidence_cuda.cpu(), confidence, rtol=0, atol=atol * confidence.amax().item())


@pytest.mark.compile
@pytest.mark.slow
@pytest.mark.parametrize(("dtype", "shape"), [(torch.float32, (2, 2, 70, 66)), (torch.float64, (1, 3, 66, 75))])
def test_compile(device: torch.device, dtype: torch.dtype, shape: tuple[int, int, int, int]) -> None:
    torch._dynamo.reset()
    batch, channels, height, width = shape
    flow = rotation_flow(height, width, 3.0, batch=batch, device=device, dtype=dtype)
    frame0, frame1 = make_pair(AnalyticTexture(2, channels=channels), flow)
    sigma = torch.tensor(2.5, device=device, dtype=dtype)
    eps = torch.tensor(0.01, device=device, dtype=dtype)

    def fn(*args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        f0, f1, init, s, e = args
        options = {"levels": 1, "iterations": 2, "window": "box", "initial_flow": init, "return_confidence": True}
        return (*lucas_kanade(f0, f1, **options), lucas_kanade(f0, f1, levels=0, iterations=2, window_sigma=s, eps=e))

    args = (frame0, frame1, 0.8 * flow, sigma, eps)
    compiled = torch.compile(fn, fullgraph=True)
    # measured: flows within 6e-7 (float32) and 2e-15 (float64), confidences within 2e-7 relative
    tol = 1e-5 if dtype == torch.float32 else 1e-12
    for out, ref in zip(compiled(*args), fn(*args), strict=True):
        torch.testing.assert_close(out, ref, rtol=tol, atol=tol)


# ----------------------------------------------------------------------------------------------
# Module
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "options",
    [
        {"levels": 0, "winsize": 11, "iterations": 3, "window_sigma": 2.0, "eps": 0.01},
        {"pyr_scale": 0.7, "levels": 1, "winsize": 9, "iterations": 4, "window": "box", "eps": 0.01},
    ],
    ids=["gaussian", "box"],
)
def test_module_matches_function(device: torch.device, options: dict[str, object]) -> None:
    # at 96x112 each of these options, levels and pyr_scale included, gives another result than its default
    flow = MOTIONS["zoom"](96, 112).to(device)
    frame0, frame1 = make_pair(AnalyticTexture(1), flow, dtype=torch.float32)
    model = LucasKanade(**options, learnable=True).to(device)
    out, confidence = model(frame0, frame1, initial_flow=0.5 * flow.float(), return_confidence=True)
    ref, ref_confidence = lucas_kanade(
        frame0, frame1, initial_flow=0.5 * flow.float(), return_confidence=True, **options
    )
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(confidence, ref_confidence, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        LucasKanade(**options).to(device)(frame0, frame1), lucas_kanade(frame0, frame1, **options)
    )


def test_module_parameters() -> None:
    assert list(LucasKanade().parameters()) == []
    model = LucasKanade(learnable=True)
    assert {name for name, _ in model.named_parameters()} == {"log_window_sigma", "log_eps"}
    assert model.window_sigma.item() == pytest.approx(3.0) and model.eps.item() == pytest.approx(1e-3)
    assert "learnable=['window_sigma', 'eps']" in repr(model)
    box = LucasKanade(window="box", learnable=True, eps=0.5)
    assert [name for name, _ in box.named_parameters()] == ["log_eps"] and box.window_sigma is None
    assert box.eps.item() == pytest.approx(0.5)
    assert "window='box'" in repr(box)
    only = LucasKanade(learnable="window_sigma", window_sigma=1.5, eps=0.2)
    assert [name for name, _ in only.named_parameters()] == ["log_window_sigma"] and only.eps == 0.2
    assert "window_sigma" not in repr(only).split("learnable")[0]
    copy = LucasKanade(learnable=True)
    copy.load_state_dict(model.state_dict())
    assert "data_range=255.0" in repr(model) and "data_range=1.0" in repr(LucasKanade(data_range=1))


def test_module_data_range(device: torch.device) -> None:
    # eps is stored in units of 0-255 frames, so a learned value carries over to another data_range
    flow = rotation_flow(64, 72, 4.0, device=device) + translation_flow(64, 72, 1.2, -0.7, device=device)
    frame0, frame1 = make_pair(AnalyticTexture(7, channels=2), flow)
    model = LucasKanade(levels=1, iterations=3, eps=100.0, learnable=True).to(device)  # as if trained
    unit = LucasKanade(levels=1, iterations=3, data_range=1.0, learnable=True).to(device)
    unit.load_state_dict(model.state_dict())
    assert unit.eps.item() == pytest.approx(100.0)
    outs = []
    for m, scale in ((model, 1.0), (unit, 1 / 255)):
        out = m(scale * frame0, scale * frame1)
        outs.append((out, torch.autograd.grad(endpoint_error(out, flow).mean(), list(m.parameters()))))
    torch.testing.assert_close(outs[1][0], outs[0][0], rtol=0, atol=1e-4)
    torch.testing.assert_close(outs[1][1], outs[0][1], rtol=1e-4, atol=0)


def test_module_training(device: torch.device) -> None:
    # frames in [0, 1] where eps = 0.05 over-damps the 3 iterations and window_sigma = 1 is too small
    flow = rotation_flow(64, 72, 4.0, batch=2, device=device) + translation_flow(64, 72, 1.2, -0.7, device=device)
    frame0, frame1 = (frame / 255 for frame in make_pair(AnalyticTexture(7, channels=2), flow))
    model = LucasKanade(levels=1, iterations=3, window_sigma=1.0, eps=0.05, learnable=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mask = interior(flow, 8)
    losses = []
    for _ in range(15):
        optimizer.zero_grad()
        loss = endpoint_error(model(frame0, frame1), flow)[mask].mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    # measured: 0.55 -> 0.069 px, eps 0.05 -> 0.012, window_sigma 1 -> 3.4
    assert losses[-1] < 0.25 * losses[0]
    assert model.eps.item() < 0.025 and model.window_sigma.item() > 2


@pytest.mark.cuda
def test_mixed_precision_training_step(grad_scaler: torch.amp.GradScaler) -> None:
    # float32 frames inside autocast(float16), with a GradScaler. Under autocast, the convolutions of
    # squared 0-255 gradients used to overflow float16: NaN flows on most pixels.
    flow = rotation_flow(96, 112, 3.0, device="cuda") + translation_flow(96, 112, 2.5, -1.5, device="cuda")
    frame0, frame1 = (frame.requires_grad_() for frame in make_pair(AnalyticTexture(1), flow))
    model = LucasKanade(learnable=True).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    with torch.autocast("cuda", dtype=torch.float16):
        out = model(frame0, frame1)
        loss = endpoint_error(out, flow).mean()
    grad_scaler.scale(loss).backward()
    grad_scaler.step(optimizer)
    grad_scaler.update()
    assert out.dtype == torch.float32 and out.isfinite().all()
    assert all(tensor.grad.isfinite().all() for tensor in (frame0, frame1, *model.parameters()))
    assert grad_scaler.get_scale() == 2.0**16  # halved if the step had been skipped for non-finite gradients


# ----------------------------------------------------------------------------------------------
# Argument checks
# ----------------------------------------------------------------------------------------------


def test_validation() -> None:
    frame = torch.zeros(1, 2, 8, 9)
    with pytest.raises(ValueError, match="same shape"):
        lucas_kanade(frame, torch.zeros(1, 2, 8, 10))
    with pytest.raises(TypeError, match="floating-point"):
        lucas_kanade(frame.long(), frame.long())
    with pytest.raises(ValueError, match=r"\(B, C, H, W\)"):
        lucas_kanade(frame[0], frame[0])
    with pytest.raises(ValueError, match="initial_flow"):
        lucas_kanade(frame, frame, initial_flow=torch.zeros(1, 2, 8, 8))
    with pytest.raises(TypeError, match="initial_flow"):
        lucas_kanade(frame, frame, initial_flow=torch.zeros(1, 2, 8, 9, dtype=torch.float64))
    cases = [
        ({"pyr_scale": 1.0}, ValueError, "pyr_scale"),
        ({"pyr_scale": "half"}, TypeError, "pyr_scale"),
        ({"levels": -1}, ValueError, "levels"),
        ({"levels": 1.5}, TypeError, "levels"),
        ({"levels": True}, TypeError, "levels"),
        ({"winsize": 0}, ValueError, "winsize"),
        ({"iterations": 0}, ValueError, "iterations"),
        ({"window": "median"}, ValueError, "window"),
        ({"window": "box", "window_sigma": 2.0}, ValueError, "window_sigma"),
        ({"window_sigma": 0.0}, ValueError, "window_sigma"),
        ({"window_sigma": torch.ones(2)}, ValueError, "window_sigma"),
        ({"eps": 0.0}, ValueError, "eps"),
        ({"eps": float("nan")}, ValueError, "eps"),
        ({"eps": "small"}, TypeError, "eps"),
        ({"eps": torch.tensor(1)}, TypeError, "eps"),
        ({"data_range": 0.0}, ValueError, "data_range must be positive"),
        ({"data_range": -1}, ValueError, "data_range must be positive"),
        ({"data_range": float("inf")}, ValueError, "data_range must be finite"),
        ({"data_range": float("nan")}, ValueError, "data_range must be finite"),
        ({"data_range": "1"}, TypeError, "data_range must be a float"),
        ({"data_range": None}, TypeError, "data_range must be a float"),
        ({"data_range": False}, TypeError, "data_range must be a float"),
        ({"data_range": torch.tensor(1.0)}, TypeError, "data_range must be a float"),
    ]
    for kwargs, error, match in cases:
        with pytest.raises(error, match=match):
            lucas_kanade(frame, frame, **kwargs)
        if not any(isinstance(value, torch.Tensor) for value in kwargs.values()):
            with pytest.raises(error, match=match):
                LucasKanade(**kwargs)


def test_module_validation() -> None:
    with pytest.raises(ValueError, match="among"):
        LucasKanade(learnable=["poly_sigma"])
    with pytest.raises(ValueError, match="window_sigma"):
        LucasKanade(window="box", learnable=["window_sigma"])
    with pytest.raises(ValueError, match="window_sigma"):
        LucasKanade(winsize=1, learnable="window_sigma")
    with pytest.raises(TypeError, match="learnable"):
        LucasKanade(learnable=3)
    with pytest.raises(TypeError, match="eps"):
        LucasKanade(eps=torch.tensor(1.0))
    with pytest.raises(TypeError, match="data_range"):
        LucasKanade(data_range=torch.tensor(1.0))


@pytest.mark.cuda
def test_device_mismatch() -> None:
    frame = torch.zeros(1, 1, 8, 9)
    with pytest.raises(ValueError, match="device"):
        lucas_kanade(frame, frame.cuda())
    with pytest.raises(ValueError, match="device"):
        lucas_kanade(frame, frame, initial_flow=torch.zeros(1, 2, 8, 9, device="cuda"))

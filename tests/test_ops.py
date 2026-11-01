"""Tests of the internal building blocks, mostly against OpenCV, and of the autocast guard of the public functions."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import torch
from synthetic import AnalyticTexture, image_from_cv, image_to_cv, make_pair, rotation_flow, smooth_texture
from torch.autograd import gradcheck

import opticalflow_torch
from opticalflow_torch import (
    Farneback,
    LucasKanade,
    _ops,
    consistency,
    endpoint_error,
    farneback,
    flow_to_rgb,
    interpolate_frames,
    lucas_kanade,
    photometric_error,
    resize_flow,
    warp,
)

cv2 = pytest.importorskip("cv2")

EPS32 = float(np.finfo(np.float32).eps)
EPS64 = float(np.finfo(np.float64).eps)
# float32 results of 0-255 images computed by OpenCV and by torch differ by a few units in the
# last place from the order of operations: an ulp at 255 is 1.5e-5.
ULP255 = 255 * EPS32 / 2
CV_BORDERS = {"reflect101": cv2.BORDER_REFLECT_101, "replicate": cv2.BORDER_REPLICATE, "zeros": cv2.BORDER_CONSTANT}


def texture_cv(height: int, width: int, seed: int = 0) -> np.ndarray:
    return image_to_cv(smooth_texture(height, width, seed=seed))


# ----------------------------------------------------------------------------------------------
# Checks and parameters
# ----------------------------------------------------------------------------------------------


def test_check_image_and_frames() -> None:
    x = torch.zeros(1, 1, 4, 5)
    _ops.check_frames(x, x.clone())
    with pytest.raises(TypeError, match="torch.Tensor"):
        _ops.check_image(np.zeros((1, 1, 4, 5)))
    with pytest.raises(ValueError, match=r"\(B, C, H, W\)"):
        _ops.check_image(torch.zeros(4, 5))
    with pytest.raises(TypeError, match="floating-point"):
        _ops.check_image(torch.zeros(1, 1, 4, 5, dtype=torch.int64))
    with pytest.raises(ValueError, match="same shape"):
        _ops.check_frames(x, torch.zeros(1, 1, 4, 6))
    with pytest.raises(TypeError, match="same dtype"):
        _ops.check_frames(x, x.double())


@pytest.mark.cuda
def test_check_frames_device_mismatch() -> None:
    x = torch.zeros(1, 1, 4, 5)
    with pytest.raises(ValueError, match="same device"):
        _ops.check_frames(x, x.cuda())
    with pytest.raises(ValueError, match="device of the frames"):
        _ops.check_flow(torch.zeros(1, 2, 4, 5, device="cuda"), x)


def test_check_flow() -> None:
    frames = torch.zeros(2, 3, 4, 5)
    _ops.check_flow(torch.zeros(2, 2, 4, 5), frames)
    _ops.check_flow(torch.zeros(7, 2, 9, 9))
    with pytest.raises(ValueError, match=r"\(B, 2, H, W\)"):
        _ops.check_flow(torch.zeros(2, 3, 4, 5))
    with pytest.raises(ValueError, match=r"\(2, 2, 4, 5\)"):
        _ops.check_flow(torch.zeros(1, 2, 4, 5), frames)
    with pytest.raises(ValueError, match=r"\(2, 2, 4, 5\)"):
        _ops.check_flow(torch.zeros(2, 2, 4, 6), frames)
    with pytest.raises(TypeError, match="dtype of the frames"):
        _ops.check_flow(torch.zeros(2, 2, 4, 5, dtype=torch.float64), frames)
    with pytest.raises(TypeError, match="init_flow must have the dtype of the image"):
        _ops.check_flow(torch.zeros(2, 2, 4, 5, dtype=torch.float64), frames, "init_flow", "image")
    with pytest.raises(ValueError, match="to match the image"):
        _ops.check_flow(torch.zeros(1, 2, 4, 5), frames, like_name="image")
    with pytest.raises(TypeError, match="floating-point"):
        _ops.check_flow(torch.zeros(2, 2, 4, 5, dtype=torch.int32))


def test_check_size() -> None:
    assert _ops.check_size((3, 4)) == (3, 4)
    assert _ops.check_size([np.int64(3), 4]) == (3, 4)
    for bad in [(3,), (3, 4, 5), (0, 4), (3, -1), (3.0, 4), 5]:
        with pytest.raises(ValueError):
            _ops.check_size(bad)


def test_as_scalar(device: torch.device) -> None:
    like = torch.zeros(2, device=device, dtype=torch.float64)
    out = _ops.as_scalar(1.5, like)
    assert out.shape == () and out.dtype == torch.float64 and out.device == like.device and out.item() == 1.5
    sigma = torch.tensor([2.0], requires_grad=True)  # float32 on CPU, converted to float64 on device
    out = _ops.as_scalar(sigma, like)
    assert out.shape == () and out.dtype == torch.float64 and out.device == like.device
    (3 * out).backward()
    assert sigma.grad.item() == 3.0
    with pytest.raises(ValueError, match="scalar"):
        _ops.as_scalar(torch.ones(2), like)
    with pytest.raises(TypeError):
        _ops.as_scalar("1.0", like)


# ----------------------------------------------------------------------------------------------
# Kernels
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("ksize", [1, 3, 5, 7, 9, 11, 13, 21, 79])
@pytest.mark.parametrize("sigma", [0.0, -1.0, 0.3, 0.5, 1.1, 2.0, 3.5, 7.5, 15.5])
def test_cv_gaussian_kernel_matches_opencv(ksize: int, sigma: float) -> None:
    out32 = _ops.cv_gaussian_kernel(ksize, sigma, dtype=torch.float32).numpy()
    np.testing.assert_array_equal(out32, cv2.getGaussianKernel(ksize, sigma, ktype=cv2.CV_32F).ravel())
    # OpenCV's software exp differs from the C library's by a few ulp: measured at most 1.1e-16
    out64 = _ops.cv_gaussian_kernel(ksize, sigma, dtype=torch.float64).numpy()
    ref64 = cv2.getGaussianKernel(ksize, sigma, ktype=cv2.CV_64F).ravel()
    np.testing.assert_allclose(out64, ref64, rtol=0, atol=2 * EPS64)


def test_cv_gaussian_kernel_rejects_even_size() -> None:
    with pytest.raises(ValueError, match="odd"):
        _ops.cv_gaussian_kernel(4, 1.0)


@pytest.mark.parametrize("radius", [0, 1, 3, 10])
@pytest.mark.parametrize("sigma", [0.4, 1.1, 3.0])
def test_gaussian_kernel1d_matches_opencv(radius: int, sigma: float) -> None:
    out = _ops.gaussian_kernel1d(sigma, radius, dtype=torch.float64).numpy()
    ref = cv2.getGaussianKernel(2 * radius + 1, sigma, ktype=cv2.CV_64F).ravel()
    np.testing.assert_allclose(out, ref, rtol=0, atol=2 * EPS64)


def test_gaussian_kernel1d_batched_sigma(device: torch.device) -> None:
    sigma = torch.tensor([0.5, 1.0, 2.0], device=device, dtype=torch.float64)
    out = _ops.gaussian_kernel1d(sigma, 4)
    assert out.shape == (3, 9) and out.device == sigma.device
    torch.testing.assert_close(out.sum(-1), torch.ones(3, device=device, dtype=torch.float64))
    for i in range(3):
        torch.testing.assert_close(out[i], _ops.gaussian_kernel1d(sigma[i], 4))
    with pytest.raises(ValueError, match="positive"):
        _ops.gaussian_kernel1d(0.0, 3)
    with pytest.raises(ValueError, match="radius"):
        _ops.gaussian_kernel1d(1.0, -1)


def test_gaussian_kernel1d_gradcheck(device: torch.device) -> None:
    sigma = torch.tensor(1.3, device=device, dtype=torch.float64, requires_grad=True)
    assert gradcheck(lambda s: _ops.gaussian_kernel1d(s, 4), (sigma,))
    sigmas = torch.tensor([0.7, 2.5], device=device, dtype=torch.float64, requires_grad=True)
    assert gradcheck(lambda s: _ops.gaussian_kernel1d(s, 3), (sigmas,))


# ----------------------------------------------------------------------------------------------
# Borders and separable filtering
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("border", list(CV_BORDERS))
@pytest.mark.parametrize(("shape", "padding"), [((6, 7), (2, 3, 1, 4)), ((3, 4), (9, 7, 8, 5)), ((1, 5), (2, 2, 3, 3))])
def test_pad_border_matches_opencv(border: str, shape: tuple[int, int], padding: tuple[int, int, int, int]) -> None:
    image = np.arange(1, shape[0] * shape[1] + 1, dtype=np.float32).reshape(shape)
    left, right, top, bottom = padding
    ref = cv2.copyMakeBorder(image, top, bottom, left, right, CV_BORDERS[border], value=0)
    out = _ops.pad_border(torch.from_numpy(image), padding, border).numpy()
    np.testing.assert_array_equal(out, ref)
    with pytest.raises(ValueError, match="non-negative"):
        _ops.pad_border(torch.from_numpy(image), (1, -1, 0, 0), border)


@pytest.mark.parametrize("shape", [(37, 53), (64, 64), (101, 77), (5, 7), (3, 2)])
@pytest.mark.parametrize(
    ("ksize", "sigma"), [(3, 0), (5, 0), (9, 0), (11, 0), (3, 0.5), (9, 1.5), (19, 3.5), (39, 7.5)]
)
def test_gaussian_blur_matches_opencv(device: torch.device, shape: tuple[int, int], ksize: int, sigma: float) -> None:
    image = texture_cv(*shape)
    ref = cv2.GaussianBlur(image, (ksize, ksize), sigma, sigma)
    kernel = _ops.cv_gaussian_kernel(ksize, sigma, device=device)
    out = image_to_cv(_ops.separable_filter(image_from_cv(image, device=device), kernel, kernel))
    np.testing.assert_allclose(out, ref, rtol=0, atol=8 * ULP255)  # measured at most 6 ulp


@pytest.mark.parametrize("border", list(CV_BORDERS))
@pytest.mark.parametrize(("shape", "kx_size", "ky_size"), [((41, 30), 7, 5), ((4, 6), 15, 11), ((23, 1), 3, 9)])
def test_separable_filter_matches_opencv(
    device: torch.device, border: str, shape: tuple[int, int], kx_size: int, ky_size: int
) -> None:
    rng = np.random.default_rng(1)
    image = texture_cv(*shape)
    kx = rng.standard_normal(kx_size).astype(np.float32)
    ky = rng.standard_normal(ky_size).astype(np.float32)
    ref = cv2.sepFilter2D(image, -1, kx, ky, borderType=CV_BORDERS[border])
    out = _ops.separable_filter(
        image_from_cv(image, device=device), torch.from_numpy(kx), torch.from_numpy(ky), border=border
    )
    # kernels of both signs: rounding errors scale with sum|kx| sum|ky| max|image|
    tol = 4 * EPS32 * np.abs(kx).sum() * np.abs(ky).sum() * image.max()
    np.testing.assert_allclose(image_to_cv(out), ref, rtol=0, atol=tol)


def test_separable_filter_per_channel_kernels(device: torch.device) -> None:
    x = smooth_texture(20, 17, batch=2, channels=3, device=device)
    kx = torch.rand(3, 5, device=device)
    ky = torch.rand(3, 7, device=device)
    out = _ops.separable_filter(x, kx, ky, border="replicate")
    for c in range(3):
        ref = _ops.separable_filter(x[:, c : c + 1], kx[c], ky[c], border="replicate")
        torch.testing.assert_close(out[:, c : c + 1], ref)
    only_x = _ops.separable_filter(x, kx, None)
    only_y = _ops.separable_filter(x, None, ky)
    torch.testing.assert_close(_ops.separable_filter(only_x, None, ky), _ops.separable_filter(x, kx, ky))
    torch.testing.assert_close(_ops.separable_filter(only_y, kx, None), _ops.separable_filter(x, kx, ky))
    with pytest.raises(ValueError, match="odd"):
        _ops.separable_filter(x, torch.ones(4, device=device), None)
    with pytest.raises(ValueError, match="channels"):
        _ops.separable_filter(x, torch.ones(2, 3, device=device), None)
    with pytest.raises(ValueError, match="border"):
        _ops.separable_filter(x, kx, None, border="reflect")


@pytest.mark.parametrize("border", list(CV_BORDERS))
def test_separable_filter_gradcheck(device: torch.device, border: str) -> None:
    kwargs = {"device": device, "dtype": torch.float64, "requires_grad": True}
    x = torch.rand(2, 2, 5, 6, **kwargs)
    kx = torch.rand(5, **kwargs)
    ky = torch.rand(2, 3, **kwargs)
    assert gradcheck(lambda x, kx, ky: _ops.separable_filter(x, kx, ky, border=border), (x, kx, ky))


@pytest.mark.parametrize("channels", [1, 2])
def test_separable_filter_is_full_precision(
    device: torch.device, channels: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    # guards against reduced-precision (TF32) convolutions on CUDA: relative error ~1e-3. With one
    # channel cuDNN computes the kernel gradient in TF32 by default (as documented; ~5e-4 measured
    # at this shape), so the test turns TF32 off; with two the depthwise kernels must not use it.
    if channels == 1:
        monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    x = smooth_texture(256, 256, channels=channels, device=device)
    kx, ky = torch.rand(21, device=device), torch.rand(21, device=device)
    kx, ky = kx / kx.sum(), ky / ky.sum()
    grad_out = torch.randn_like(x)

    def run(*inputs: torch.Tensor) -> list[torch.Tensor]:
        inputs = [t.detach().requires_grad_() for t in inputs]
        out = _ops.separable_filter(*inputs)
        return [out.detach(), *torch.autograd.grad(out, inputs, grad_out.to(out.dtype))]

    results = run(x, kx, ky)
    references = run(x.double(), kx.double(), ky.double())
    assert (results[0].double() - references[0]).abs().max().item() < 1e-6 * 255
    for grad, ref in zip(results[1:], references[1:], strict=True):  # x, kx, ky: measured at most 5e-6
        assert (grad.double() - ref).abs().max().item() < 2e-5 * ref.abs().max().item()


# ----------------------------------------------------------------------------------------------
# Resizing
# ----------------------------------------------------------------------------------------------

RESIZE_CASES = [
    ((64, 64), (32, 32)),
    ((101, 77), (51, 39)),
    ((101, 77), (50, 38)),
    ((100, 100), (37, 61)),
    ((96, 90), (32, 30)),
    ((97, 64), (37, 64)),
    ((1000, 777), (500, 389)),
    ((33, 47), (80, 95)),
    ((50, 40), (50, 77)),
    ((40, 60), (20, 90)),
    ((7, 5), (3, 2)),
    ((1, 20), (3, 7)),
    # INTER_AREA's two-tap mode where (float)fx wraps 1 - 4e-15 to 0: row 25 and column 27
    ((22, 56), (50, 36)),
    # a single row or column: OpenCV 5 keeps OpenCV 4's float coordinates (off by 0.03 in double)
    ((1, 4000), (1, 1300)),
    ((4000, 1), (1300, 1)),
    ((1, 4000), (3, 1300)),
    # INTER_AREA drops partial overlaps up to 1e-3 pixels: on the right (2.001), on the left (1.999)
    ((2001, 40), (1000, 20)),
    ((40, 2003), (20, 1002)),
]


@pytest.mark.parametrize("interpolation", ["linear", "area"])
@pytest.mark.parametrize(("shape", "size"), RESIZE_CASES)
def test_resize_matches_opencv(
    device: torch.device, interpolation: str, shape: tuple[int, int], size: tuple[int, int]
) -> None:
    image = texture_cv(*shape)
    flag = cv2.INTER_LINEAR if interpolation == "linear" else cv2.INTER_AREA
    ref = cv2.resize(image, size[::-1], interpolation=flag)
    out = _ops.resize(torch.from_numpy(image).to(device), size, interpolation).cpu().numpy()
    # measured at most 2 ulp (linear) and 4 ulp (area by an integer factor other than 2)
    np.testing.assert_allclose(out, ref, rtol=0, atol=6 * ULP255)


@pytest.mark.parametrize("axis", [-1, -2])
@pytest.mark.parametrize(
    ("interpolation", "lines"), [("linear", (2, 2)), ("linear", (1, 1)), ("area", (2, 2)), ("area", (2, 3))]
)
def test_resize_lengths_match_opencv(interpolation: str, lines: tuple[int, int], axis: int) -> None:
    # every pair of lengths in 1..64 along one axis, with the other axis of length lines[0] -> lines[1];
    # 2 -> 3 makes OpenCV use its two-tap area mode on a shrinking axis. Random values make a tap
    # on the wrong pixel stand out (errors of tens to hundreds)
    rng = np.random.default_rng(0)
    flag = cv2.INTER_LINEAR if interpolation == "linear" else cv2.INTER_AREA
    failures = []
    for n_in in range(1, 65):
        image = (255 * rng.random((lines[0], n_in))).astype(np.float32)
        image = image if axis == -1 else np.ascontiguousarray(image.T)
        for n_out in range(1, 65):
            size = (lines[1], n_out) if axis == -1 else (n_out, lines[1])
            ref = cv2.resize(image, size[::-1], interpolation=flag).reshape(size)
            out = _ops.resize(torch.from_numpy(image), size, interpolation).numpy()
            if np.abs(out - ref).max() > 6 * ULP255:
                failures.append((n_in, n_out))
    assert not failures


@pytest.mark.parametrize(
    ("shape", "size"),
    [((33, 39), (20, 20)), ((20, 20), (33, 39)), ((22, 56), (50, 36)), ((1, 4000), (3, 1300)), ((4000, 2), (1300, 2))],
)
def test_resize_float64_matches_opencv(device: torch.device, shape: tuple[int, int], size: tuple[int, int]) -> None:
    # as documented, float64 area weights stay in double where OpenCV rounds them to float
    image = 255 * np.random.default_rng(0).random(shape)
    for interpolation, flag, atol in (("linear", cv2.INTER_LINEAR, 1e-9), ("area", cv2.INTER_AREA, 2e-5)):
        ref = cv2.resize(image, size[::-1], interpolation=flag).reshape(size)
        out = _ops.resize(torch.from_numpy(image).to(device), size, interpolation).cpu().numpy()
        np.testing.assert_allclose(out, ref, rtol=0, atol=atol)


@pytest.mark.parametrize(("shape", "size"), [((64, 80), (32, 40)), ((101, 77), (51, 39)), ((480, 640), (15, 20))])
def test_resize_area_of_flow_matches_opencv(
    device: torch.device, shape: tuple[int, int], size: tuple[int, int]
) -> None:
    # Farneback shrinks initial flows with INTER_AREA, channel by channel
    flow = 20 * (image_to_cv(smooth_texture(*shape, channels=2)) / 255 - 0.5)
    ref = cv2.resize(flow, size[::-1], interpolation=cv2.INTER_AREA)
    out = _ops.resize(torch.from_numpy(flow).permute(2, 0, 1).to(device), size, "area")
    np.testing.assert_allclose(out.permute(1, 2, 0).cpu().numpy(), ref, rtol=0, atol=6 * 10 * EPS32 / 2)


def test_resize_same_size_and_errors() -> None:
    x = torch.rand(2, 3, 8, 9)
    assert _ops.resize(x, (8, 9)) is x
    assert _ops.resize(x, (5, 4), "area").shape == (2, 3, 5, 4)
    with pytest.raises(ValueError, match="interpolation"):
        _ops.resize(x, (4, 4), "cubic")
    with pytest.raises(ValueError):
        _ops.resize(x, (0, 4))
    with pytest.raises(TypeError):
        _ops.resize(torch.ones(4, 4, dtype=torch.int64), (2, 2))
    with pytest.raises(ValueError, match=r"\(\.\.\., H, W\)"):
        _ops.resize(torch.ones(4), (2, 2))


@pytest.mark.parametrize("interpolation", ["linear", "area"])
@pytest.mark.parametrize("size", [(3, 4), (11, 13), (4, 13)])
@pytest.mark.parametrize("shape", [(7, 9), (1, 9)])
def test_resize_gradcheck(
    device: torch.device, interpolation: str, size: tuple[int, int], shape: tuple[int, int]
) -> None:
    x = torch.rand(2, 1, *shape, device=device, dtype=torch.float64, requires_grad=True)
    assert gradcheck(lambda x: _ops.resize(x, size, interpolation), (x,))


# ----------------------------------------------------------------------------------------------
# Farneback's pyramid
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ((64, 200, 0.5, 5), [(64, 200), (32, 100)]),
        ((63, 200, 0.5, 5), [(63, 200)]),  # 31.5 < 32: the unrounded size is compared
        ((101, 75, 0.5, 5), [(101, 75), (50, 38)]),  # cvRound ties to even: 50.5 -> 50, 37.5 -> 38
        ((480, 640, 0.5, 5), [(480, 640), (240, 320), (120, 160), (60, 80)]),
        ((480, 640, 0.5, 2), [(480, 640), (240, 320), (120, 160)]),
        ((100, 100, 0.8, 10), [(100, 100), (80, 80), (64, 64), (51, 51), (41, 41), (33, 33)]),
        ((480, 640, 0.5, 0), [(480, 640)]),
        ((20, 20, 0.5, 3), [(20, 20)]),
    ],
)
def test_pyramid_levels(args: tuple, expected: list[tuple[int, int]]) -> None:
    levels = _ops.pyramid_levels(*args)
    assert [(level.height, level.width) for level in levels] == expected
    scale = 1.0
    for level in levels:
        assert level.scale == scale  # repeated products, bit for bit as OpenCV
        scale *= args[2]


def test_pyramid_levels_errors() -> None:
    for pyr_scale in (0.0, 1.0, 1.5, -0.5):
        with pytest.raises(ValueError, match="pyr_scale"):
            _ops.pyramid_levels(64, 64, pyr_scale, 3)
    with pytest.raises(ValueError, match="levels"):
        _ops.pyramid_levels(64, 64, 0.5, -1)


@pytest.mark.parametrize(
    ("shape", "pyr_scale"), [((64, 200), 0.5), ((63, 200), 0.5), ((96, 130), 0.5), ((90, 90), 0.8)]
)
def test_pyramid_level_count_matches_opencv(shape: tuple[int, int], pyr_scale: float) -> None:
    # OpenCV's Farneback gives identical flows for all level counts at or above its cap
    frame0 = texture_cv(*shape, seed=1)
    frame1 = np.roll(frame0, (1, 2), axis=(0, 1))
    count = len(_ops.pyramid_levels(*shape, pyr_scale, 10)) - 1

    def flow(levels: int) -> np.ndarray:
        return cv2.calcOpticalFlowFarneback(frame0, frame1, None, pyr_scale, levels, 9, 2, 5, 1.1, 0)

    np.testing.assert_array_equal(flow(count), flow(count + 3))
    if count > 0:
        assert not np.array_equal(flow(count - 1), flow(count))


@pytest.mark.parametrize(
    ("shape", "pyr_scale", "levels"), [((101, 77), 0.5, 5), ((160, 230), 0.5, 5), ((128, 96), 0.8, 4)]
)
def test_pyramid_level_matches_opencv(
    device: torch.device, shape: tuple[int, int], pyr_scale: float, levels: int
) -> None:
    image = texture_cv(*shape)
    pyramid = _ops.build_pyramid(image_from_cv(image, device=device), pyr_scale, levels)
    sizes = _ops.pyramid_levels(*shape, pyr_scale, levels)
    assert len(pyramid) == len(sizes) > 1
    for level, out in zip(sizes, pyramid, strict=True):
        # the CPU path of FarnebackOpticalFlowImpl::calc
        sigma = (1 / level.scale - 1) * 0.5
        ksize = max(round(sigma * 5) | 1, 3)
        ref = cv2.resize(cv2.GaussianBlur(image, (ksize, ksize), sigma, sigma), (level.width, level.height))
        np.testing.assert_allclose(image_to_cv(out), ref, rtol=0, atol=8 * ULP255)
        torch.testing.assert_close(out, _ops.pyramid_level(image_from_cv(image, device=device), level))


def test_build_pyramid_batched_channels(device: torch.device) -> None:
    x = smooth_texture(70, 90, batch=2, channels=3, device=device, dtype=torch.float64)
    pyramid = _ops.build_pyramid(x, 0.5, 3)
    assert [p.shape for p in pyramid] == [(2, 3, 70, 90), (2, 3, 35, 45)]
    single = _ops.build_pyramid(x[1:, 2:], 0.5, 3)
    for p, s in zip(pyramid, single, strict=True):
        torch.testing.assert_close(p[1:, 2:], s, rtol=0, atol=1e-12)


def test_build_pyramid_gradcheck(device: torch.device) -> None:
    x = torch.rand(1, 1, 12, 14, device=device, dtype=torch.float64, requires_grad=True)

    def fn(x: torch.Tensor) -> torch.Tensor:
        pyramid = _ops.build_pyramid(x, 0.5, 2, min_size=3)
        assert [p.shape[-2:] for p in pyramid] == [(12, 14), (6, 7), (3, 4)]
        return torch.cat([p.flatten() for p in pyramid])

    assert gradcheck(fn, (x,))


# ----------------------------------------------------------------------------------------------
# Devices and compilation
# ----------------------------------------------------------------------------------------------


@pytest.mark.cuda
def test_cpu_cuda_agree() -> None:
    x = smooth_texture(101, 77, batch=2, channels=2)
    kernel = _ops.cv_gaussian_kernel(9, 1.5)
    for fn in (
        lambda x, k: _ops.separable_filter(x, k, k),
        lambda x, k: _ops.resize(x, (51, 39)),
        lambda x, k: _ops.resize(x, (40, 30), "area"),
        lambda x, k: _ops.resize(x, (150, 100), "area"),
        lambda x, k: _ops.build_pyramid(x, 0.5, 5)[-1],
    ):
        torch.testing.assert_close(fn(x.cuda(), kernel.cuda()).cpu(), fn(x, kernel), rtol=0, atol=8 * ULP255)


@pytest.mark.compile
def test_compile_pyramid(device: torch.device) -> None:
    x = smooth_texture(80, 72, batch=2, device=device)

    def fn(x: torch.Tensor) -> list[torch.Tensor]:
        pyramid = _ops.build_pyramid(x, 0.5, 3)
        return [*pyramid, _ops.resize(pyramid[0], (20, 18), "area")]

    compiled = torch.compile(fn, fullgraph=True)
    for out, ref in zip(compiled(x), fn(x), strict=True):
        torch.testing.assert_close(out, ref, rtol=0, atol=8 * ULP255)


# ----------------------------------------------------------------------------------------------
# Autocast
# ----------------------------------------------------------------------------------------------

AUTOCAST = [
    pytest.param("cpu", torch.bfloat16, id="cpu-bfloat16"),
    pytest.param("cuda", torch.float16, id="cuda-float16", marks=pytest.mark.cuda),
    pytest.param("cuda", torch.bfloat16, id="cuda-bfloat16", marks=pytest.mark.cuda),
]


def public_calls(device: str) -> dict[str, Callable[[], object]]:
    """A call of each public name on float32 inputs, the frames in 0-255."""
    flow = rotation_flow(64, 72, 3.0, device=device) + 2.0
    frame0, frame1 = make_pair(AnalyticTexture(3, channels=2), flow)
    options = {"levels": 1, "iterations": 3, "learnable": True}
    return {
        "Farneback": lambda: Farneback(window="gaussian", **options).to(device)(frame0, frame1, 0.5 * flow, True),
        "LucasKanade": lambda: LucasKanade(window="box", **options).to(device)(frame0, frame1, 0.5 * flow, True),
        "consistency": lambda: consistency(flow, 0.1 - flow),
        "endpoint_error": lambda: endpoint_error(flow, 0.9 * flow),
        "farneback": lambda: farneback(frame0, frame1, return_confidence=True),
        "flow_to_rgb": lambda: flow_to_rgb(flow),
        "interpolate_frames": lambda: interpolate_frames(frame0, frame1, flow, [0.3, 0.7], match_distribution=True),
        "lucas_kanade": lambda: lucas_kanade(frame0, frame1, return_confidence=True),
        "photometric_error": lambda: photometric_error(frame0, frame1, flow),
        "resize_flow": lambda: resize_flow(flow, (40, 90), mode="bicubic"),
        "warp": lambda: warp(frame1, flow, mode="bicubic"),
    }


def test_public_calls_cover_the_api() -> None:
    assert sorted(public_calls("cpu")) == sorted(opticalflow_torch.__all__)


@pytest.mark.parametrize(("device_type", "dtype"), AUTOCAST)
@pytest.mark.parametrize("name", opticalflow_torch.__all__)
def test_autocast_changes_nothing(name: str, device_type: str, dtype: torch.dtype) -> None:
    # Autocast used to run the convolutions of the flow methods in half precision. On these frames
    # lucas_kanade gave NaN under CUDA float16 and flows 0.12 px off under bfloat16, farneback flows
    # up to 0.17 px off, and the Farneback module NaN on 61% of the pixels under CUDA float16. The
    # other functions were not affected.
    call = public_calls(device_type)[name]
    expected = call()
    with torch.autocast(device_type, dtype=dtype):
        out = call()
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize(("device_type", "dtype"), AUTOCAST)
@pytest.mark.parametrize("module", [Farneback, LucasKanade])
def test_autocast_half_module(module: type[torch.nn.Module], device_type: str, dtype: torch.dtype) -> None:
    # A module in the autocast dtype. Its learnable values used to be computed before the guard, and
    # CUDA autocast runs exp in float32: with bfloat16 parameters eps went from 0.300781 to 0.300254,
    # and the flows moved by up to 0.011 px.
    flow = rotation_flow(64, 72, 3.0, device=device_type) + 2.0
    frame0, frame1 = make_pair(AnalyticTexture(3), flow)
    model = module(levels=1, iterations=3, eps=0.3, learnable=True).to(device_type, dtype)

    def step(autocast: bool) -> tuple[torch.Tensor, ...]:
        model.zero_grad()
        with torch.autocast(device_type, dtype=dtype, enabled=autocast):
            out = model(frame0, frame1)
        out.square().mean().backward()
        return out, *(p.grad for p in model.parameters())

    torch.testing.assert_close(step(True), step(False), rtol=0, atol=0)


@pytest.mark.compile
def test_autocast_under_compile(device: torch.device) -> None:
    torch._dynamo.reset()
    flow = rotation_flow(40, 44, 3.0, device=device) + 1.0
    frame0, frame1 = make_pair(AnalyticTexture(3), flow)

    def fn(f0: torch.Tensor, f1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        options = {"levels": 0, "iterations": 2, "winsize": 7}
        return farneback(f0, f1, **options), lucas_kanade(f0, f1, **options)

    expected = fn(frame0, frame1)
    compiled = torch.compile(fn, fullgraph=True)
    with torch.autocast(device.type, dtype=torch.float16 if device.type == "cuda" else torch.bfloat16):
        out = compiled(frame0, frame1)
    # Inductor's rounding moves the flows by up to 7e-5 px, with or without autocast; autocast used to
    # move them by 0.02-0.28 px
    torch.testing.assert_close(out, expected, rtol=0, atol=5e-4)

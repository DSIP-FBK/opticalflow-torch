"""Tests of warp and resize_flow."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from synthetic import AnalyticTexture, make_pair, rotation_flow, smooth_texture, translation_flow, zoom_flow
from torch.autograd import gradcheck

from opticalflow_torch import resize_flow, warp

MODES = ["bilinear", "nearest", "bicubic"]
PADDINGS = ["border", "zeros", "reflection"]


def rounding_tol(image: torch.Tensor, mode: str = "bilinear") -> float:
    """Bound on float rounding errors of warp: grid_sample's normalised coordinates are off by
    about eps * size pixels, which moves values by up to that times the largest pixel difference
    (bilinear), or a few times more (bicubic: four taps, kernel slopes up to 1.25)."""
    eps = torch.finfo(image.dtype).eps
    step = max((image.diff(dim=-1).abs().amax().item(), image.diff(dim=-2).abs().amax().item()))
    slope = 4 if mode == "bicubic" else 1
    return 4 * eps * max(image.shape[-2:]) * step * slope + 4 * eps * image.abs().amax().item()


def reflect101(coord: np.ndarray, size: int) -> np.ndarray:
    if size == 1:
        return np.zeros_like(coord)
    period = 2 * (size - 1)
    coord = np.abs(coord) % period
    return np.where(coord > size - 1, period - coord, coord)


def keys_cubic(t: np.ndarray, a: float = -0.75) -> np.ndarray:
    """Keys' cubic convolution kernel, as in PyTorch's bicubic interpolation."""
    t = np.abs(t)
    inner = ((a + 2) * t - (a + 3)) * t * t + 1
    outer = ((a * t - 5 * a) * t + 8 * a) * t - 4 * a
    return np.where(t <= 1, inner, np.where(t < 2, outer, 0.0))


def warp_reference(image: torch.Tensor, flow: torch.Tensor, padding: str, mode: str = "bilinear") -> torch.Tensor:
    """Sampling in pixel coordinates, written independently of grid_sample, in float64.

    Bilinear and nearest pad the sampling position (clamped or mirrored, nearest rounding half to
    even); bicubic pads each of its 4 x 4 taps, as grid_sample does.
    """
    img = image.double().cpu().numpy()
    fl = flow.double().cpu().numpy()
    batch, _, height, width = img.shape
    x = np.arange(width) + fl[:, 0]
    y = np.arange(height)[:, None] + fl[:, 1]

    def axis_taps(coord: np.ndarray, size: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """(tap index, weight) pairs along one axis."""
        if mode == "bicubic":
            base = np.floor(coord)
            return [(base + k, keys_cubic(coord - base - k)) for k in (-1, 0, 1, 2)]
        if padding == "border":
            coord = np.clip(coord, 0, size - 1)
        elif padding == "reflection":
            coord = reflect101(coord, size)
        if mode == "nearest":
            return [(np.rint(coord), np.ones_like(coord))]
        base = np.floor(coord)
        return [(base, 1 - (coord - base)), (base + 1, coord - base)]

    def source(index: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
        """Source pixel of each tap index and whether it counts (zeros padding)."""
        index = index.astype(int)
        if padding == "reflection":
            return reflect101(index, size).astype(int), np.ones(index.shape, bool)
        return np.clip(index, 0, size - 1), (padding == "border") | ((index >= 0) & (index < size))

    out = np.zeros_like(img)
    b = np.arange(batch)[:, None, None]
    for yi, wy in axis_taps(y, height):
        for xi, wx in axis_taps(x, width):
            (rows, valid_y), (cols, valid_x) = source(yi, height), source(xi, width)
            values = img[b, :, rows, cols]  # (B, H, W, C)
            out += np.moveaxis(values * (wx * wy * valid_x * valid_y)[..., None], -1, 1)
    return torch.from_numpy(out)


# ----------------------------------------------------------------------------------------------
# warp
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("padding_mode", PADDINGS)
def test_identity(device: torch.device, dtype: torch.dtype, padding_mode: str) -> None:
    image = smooth_texture(61, 83, batch=2, channels=3, device=device, dtype=dtype)
    out = warp(image, torch.zeros(2, 2, 61, 83, device=device, dtype=dtype), padding_mode=padding_mode)
    torch.testing.assert_close(out, image, rtol=0, atol=rounding_tol(image))


@pytest.mark.parametrize("padding_mode", PADDINGS)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("shape", [(23, 31), (1, 9), (8, 1), (4, 5), (1, 1)])
def test_matches_pixel_space_reference(
    device: torch.device, mode: str, padding_mode: str, shape: tuple[int, int]
) -> None:
    # random flows reaching well outside the image exercise the padding
    height, width = shape
    image = 255 * torch.rand(2, 2, height, width, device=device, dtype=torch.float64)
    flow = 6 * torch.randn(2, 2, height, width, device=device, dtype=torch.float64)
    out = warp(image, flow, mode=mode, padding_mode=padding_mode)
    torch.testing.assert_close(out.cpu(), warp_reference(image, flow, padding_mode, mode), rtol=0, atol=1e-9)


@pytest.mark.parametrize("padding_mode", PADDINGS)
@pytest.mark.parametrize(("dx", "dy"), [(3, -2), (-5, 0), (0, 7), (40, -1)])
def test_integer_shift(device: torch.device, dtype: torch.dtype, padding_mode: str, dx: int, dy: int) -> None:
    height, width = 32, 36
    image = smooth_texture(height, width, device=device, dtype=dtype)
    out = warp(image, translation_flow(height, width, dx, dy, device=device, dtype=dtype), padding_mode=padding_mode)
    rows, cols = torch.arange(height) + dy, torch.arange(width) + dx
    if padding_mode == "reflection":
        source_rows = torch.from_numpy(reflect101(rows.numpy(), height))
        source_cols = torch.from_numpy(reflect101(cols.numpy(), width))
    else:
        source_rows, source_cols = rows.clamp(0, height - 1), cols.clamp(0, width - 1)
    expected = image[..., source_rows.to(device), :][..., source_cols.to(device)]
    if padding_mode == "zeros":
        inside = ((rows >= 0) & (rows < height))[:, None] & ((cols >= 0) & (cols < width))
        expected = expected * inside.to(device)
    torch.testing.assert_close(out, expected, rtol=0, atol=rounding_tol(image))


FLOWS = {
    "translation": lambda h, w: translation_flow(h, w, 1.37, -0.62, dtype=torch.float64),
    "rotation": lambda h, w: rotation_flow(h, w, 4.0, dtype=torch.float64),
    "zoom": lambda h, w: zoom_flow(h, w, 1.06, dtype=torch.float64),
    "shrink": lambda h, w: zoom_flow(h, w, 0.95, center=(20.0, 30.0), dtype=torch.float64),
}


def inside(flow: torch.Tensor, margin: float) -> torch.Tensor:
    """Pixels whose displaced position is at least ``margin`` pixels inside the frame, shape (B, 1, H, W)."""
    height, width = flow.shape[-2:]
    x = torch.arange(width, device=flow.device) + flow[:, 0]
    y = torch.arange(height, device=flow.device)[:, None] + flow[:, 1]
    return ((x >= margin) & (x <= width - 1 - margin) & (y >= margin) & (y <= height - 1 - margin)).unsqueeze(1)


@pytest.mark.parametrize("name", list(FLOWS))
def test_affine_image_is_warped_exactly(device: torch.device, dtype: torch.dtype, name: str) -> None:
    # bilinear interpolation reproduces affine images exactly, so the warp of frame1(p) = a + b.p
    # must be frame1(p + flow(p)) up to rounding, for any flow (PyTorch's bicubic, Keys with
    # a = -0.75, only reproduces constants)
    flow64 = FLOWS[name](64, 72)
    y, x = torch.meshgrid(torch.arange(64, dtype=torch.float64), torch.arange(72, dtype=torch.float64), indexing="ij")

    def ramp(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.stack((20 + 1.7 * x - 0.9 * y, 200 - 0.4 * x - 2.1 * y), dim=-3)

    frame1 = ramp(x, y).expand(1, -1, -1, -1).to(device, dtype)
    frame0 = ramp(x + flow64[:, 0], y + flow64[:, 1]).to(device, dtype)
    flow = flow64.to(device, dtype)
    out = warp(frame1, flow)
    valid = inside(flow, margin=0).expand_as(out)
    assert valid.float().mean().item() > 0.8
    torch.testing.assert_close(out[valid], frame0[valid], rtol=0, atol=rounding_tol(frame1))


@pytest.mark.parametrize(("dx", "dy"), [(0.3, -0.45), (-2.71, 1.5), (0.5, 0.5)])
def test_subpixel_shift_of_analytic_texture(device: torch.device, dtype: torch.dtype, dx: float, dy: float) -> None:
    texture = AnalyticTexture(3, min_wavelength=12)
    flow = translation_flow(48, 56, dx, dy, device=device, dtype=dtype)
    frame0, frame1 = make_pair(texture, flow)
    out = warp(frame1, flow)
    reference = warp_reference(frame1, flow, "border").to(device, dtype)
    torch.testing.assert_close(out, reference, rtol=0, atol=rounding_tol(frame1))
    # close to the texture itself, up to the bilinear interpolation error, which is about
    # A (1 - cos(pi / 12)) ~ 0.9 for waves of amplitude A ~ 28 and wavelength 12 px at half-pixel
    # offsets; measured: mean 0.91-1.00, max 2.9-3.2. The frames differ by 6-38 on average.
    error = (out - frame0)[..., 4:-4, 4:-4].abs()
    assert error.mean().item() < 1.3 and error.amax().item() < 4.5


@pytest.mark.parametrize("name", list(FLOWS))
@pytest.mark.parametrize("mode", ["bilinear", "bicubic"])
def test_warp_recovers_frame0(device: torch.device, dtype: torch.dtype, name: str, mode: str) -> None:
    texture = AnalyticTexture(5, min_wavelength=12)
    flow = FLOWS[name](64, 72).to(device=device, dtype=dtype)
    frame0, frame1 = make_pair(texture, flow)
    out = warp(frame1, flow, mode=mode)
    valid = inside(flow, margin=2).expand_as(out)
    error = (out - frame0).abs()[valid]
    # interpolation error, see test_subpixel_shift_of_analytic_texture; measured: bilinear mean
    # 0.63-0.89, max 3.4-4.9; bicubic mean 0.50-0.57, max 2.0-3.7. The frames differ by 13-21.
    limits = {"bilinear": (1.2, 6.5), "bicubic": (0.75, 5.0)}[mode]
    assert error.mean().item() < limits[0] and error.amax().item() < limits[1]
    assert (frame1 - frame0).abs()[valid].mean().item() > 10


@pytest.mark.parametrize("padding_mode", PADDINGS)
@pytest.mark.parametrize("mode", MODES)
def test_non_finite_flow(device: torch.device, dtype: torch.dtype, mode: str, padding_mode: str) -> None:
    # grid_sample's CPU backward used to crash (SIGSEGV) on NaN positions, and its forward to
    # sample pixel (0, 0); CUDA and CPU disagreed with zeros padding
    image = smooth_texture(16, 20, channels=2, device=device, dtype=dtype).requires_grad_()
    flow = 2 * torch.randn(1, 2, 16, 20, device=device, dtype=dtype)
    bad = torch.zeros(1, 1, 16, 20, dtype=torch.bool, device=device)
    for (i, j), channels, value in [
        ((3, 4), [0], float("nan")),
        ((5, 6), [1], float("nan")),
        ((7, 8), [0, 1], float("inf")),
        ((9, 0), [1], -float("inf")),
        ((15, 19), [0, 1], float("nan")),
    ]:
        flow[0, channels, i, j] = value
        bad[0, 0, i, j] = True
    flow.requires_grad_()
    out = warp(image, flow, mode=mode, padding_mode=padding_mode)
    assert out[bad.expand_as(out)].isnan().all()
    expected = warp(image, torch.where(bad, 0.0, flow.detach()), mode=mode, padding_mode=padding_mode)
    torch.testing.assert_close(out[~bad.expand_as(out)], expected[~bad.expand_as(out)], rtol=0, atol=0)
    out.nansum().backward()
    assert image.grad.isfinite().all() and flow.grad.isfinite().all()
    assert (flow.grad[bad.expand_as(flow)] == 0).all()


@pytest.mark.parametrize("padding_mode", PADDINGS)
@pytest.mark.parametrize("mode", MODES)
def test_far_outside_flow(device: torch.device, mode: str, padding_mode: str) -> None:
    # +-3e38 overflows grid_sample's normalised coordinates in float32
    image = smooth_texture(6, 7, channels=2, device=device).requires_grad_()
    for value in (3e38, -3e38, 1e6):
        flow = torch.full((1, 2, 6, 7), value, device=device).requires_grad_()
        out = warp(image, flow, mode=mode, padding_mode=padding_mode)
        out.sum().backward()
        assert out.isfinite().all() and image.grad.isfinite().all() and flow.grad.isfinite().all()
        if padding_mode == "zeros":
            assert (out == 0).all()
        elif padding_mode == "border":
            corner = image[..., -1:, -1:] if value > 0 else image[..., :1, :1]
            torch.testing.assert_close(out, corner.detach().expand_as(out))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_is_sampled_in_float32(device: torch.device, dtype: torch.dtype) -> None:
    # grids in these dtypes cannot address pixels beyond 256 (bfloat16) or 2048 (float16): identity
    # warps used to move pixels and return values out of range
    width = 4096
    image = torch.rand(1, 2, 3, width, device=device).to(dtype).requires_grad_()
    flow = translation_flow(3, width, 2, 0, device=device, dtype=dtype)
    for mode in MODES:
        for padding_mode in PADDINGS:
            out = warp(image, flow, mode=mode, padding_mode=padding_mode)
            assert out.dtype == dtype
            # float32 coordinates are within 5e-4 px of the pixels, then the values round to dtype
            atol = torch.finfo(dtype).eps
            torch.testing.assert_close(out[..., :-2], image[..., 2:].detach(), rtol=0, atol=atol)
    out.float().sum().backward()
    assert image.grad.dtype == dtype and image.grad.isfinite().all()


def test_nearest_mode(device: torch.device) -> None:
    image = smooth_texture(20, 24, device=device)
    out = warp(image, translation_flow(20, 24, 0.3, -0.3, device=device), mode="nearest")
    torch.testing.assert_close(out, image)
    out = warp(image, translation_flow(20, 24, 0.7, 1.2, device=device), mode="nearest")
    torch.testing.assert_close(out[..., :-1, :-1], image[..., 1:, 1:])


def test_batch_independence(device: torch.device) -> None:
    image = smooth_texture(30, 40, batch=4, channels=2, device=device)
    flow = 3 * torch.randn(4, 2, 30, 40, device=device)
    for mode in ("bilinear", "bicubic", "nearest"):
        out = warp(image, flow, mode=mode)
        single = torch.cat([warp(image[i : i + 1], flow[i : i + 1], mode=mode) for i in range(4)])
        torch.testing.assert_close(out, single, rtol=0, atol=0)


@pytest.mark.parametrize("padding_mode", PADDINGS)
@pytest.mark.parametrize("mode", ["bilinear", "bicubic"])
def test_warp_gradcheck(device: torch.device, mode: str, padding_mode: str) -> None:
    image = torch.rand(2, 2, 5, 6, device=device, dtype=torch.float64, requires_grad=True)
    # non-integer positions, some outside the image: away from the kinks of the interpolation
    flow = (3 * torch.rand(2, 2, 5, 6, device=device, dtype=torch.float64) - 1.5 + 0.05).requires_grad_()
    assert gradcheck(lambda i, f: warp(i, f, mode=mode, padding_mode=padding_mode), (image, flow))


def test_warp_gradcheck_nearest(device: torch.device) -> None:
    image = torch.rand(1, 2, 5, 6, device=device, dtype=torch.float64, requires_grad=True)
    flow = 2 * torch.rand(1, 2, 5, 6, device=device, dtype=torch.float64) - 1
    assert gradcheck(lambda i: warp(i, flow, mode="nearest"), (image,))


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cpu_cuda_agree(dtype: torch.dtype) -> None:
    image = smooth_texture(50, 70, batch=2, channels=2, dtype=dtype)
    flow = 4 * torch.randn(2, 2, 50, 70, dtype=dtype)
    for mode in ("bilinear", "bicubic"):
        for padding_mode in PADDINGS:
            out = warp(image.cuda(), flow.cuda(), mode=mode, padding_mode=padding_mode).cpu()
            ref = warp(image, flow, mode=mode, padding_mode=padding_mode)
            torch.testing.assert_close(out, ref, rtol=0, atol=2 * rounding_tol(image, mode))  # two roundings


@pytest.mark.compile
@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
@pytest.mark.parametrize(("dtype", "shape"), [(torch.float32, (42, 98)), (torch.float64, (50, 99))])
def test_compile(device: torch.device, dtype: torch.dtype, shape: tuple[int, int], dynamic: bool) -> None:
    # grid_sample's own reflection, compiled for CUDA, read positions exactly on the far edge from
    # index 0 at sizes like these (depending on the dtype; other sizes with dynamic shapes); zero
    # and integer flows put them there. A dynamic height and width (as after a change of frame size)
    # also caught a miscompiled NaN guard.
    torch._dynamo.reset()  # no automatic dynamic shapes from earlier tests
    height, width = shape
    image = smooth_texture(height, width, batch=2, device=device, dtype=dtype)

    def fn(image: torch.Tensor, flow: torch.Tensor) -> list[torch.Tensor]:
        cases = [(mode, "reflection") for mode in MODES] + [("bilinear", "border"), ("bilinear", "zeros")]
        return [warp(image, flow, mode=mode, padding_mode=padding) for mode, padding in cases]

    compiled = torch.compile(fn, fullgraph=True)
    nan_flow = 2 * torch.randn(2, 2, height, width, device=device, dtype=dtype)
    nan_flow[0, 0, 5, 6] = float("nan")
    flows = [
        torch.zeros(2, 2, height, width, device=device, dtype=dtype),
        translation_flow(height, width, 3, 1 - height, batch=2, device=device, dtype=dtype),
        translation_flow(height, width, 1 - width, 2 * height - 2, batch=2, device=device, dtype=dtype),
        nan_flow,
    ]
    if dynamic:
        for tensor in (image, *flows):
            torch._dynamo.mark_dynamic(tensor, (2, 3))
    for flow in flows:
        for out, ref in zip(compiled(image, flow), fn(image, flow), strict=True):
            torch.testing.assert_close(out, ref, rtol=0, atol=rounding_tol(image, "bicubic"), equal_nan=True)
    for out in compiled(image, flows[0]):
        torch.testing.assert_close(out, image, rtol=0, atol=rounding_tol(image, "bicubic"))


def test_warp_validation() -> None:
    image = torch.zeros(2, 3, 8, 9)
    flow = torch.zeros(2, 2, 8, 9)
    with pytest.raises(ValueError, match="image"):
        warp(image[0], flow)
    with pytest.raises(TypeError, match="image"):
        warp(image.long(), flow)
    with pytest.raises(ValueError, match="flow"):
        warp(image, torch.zeros(2, 3, 8, 9))
    with pytest.raises(ValueError, match="flow"):
        warp(image, flow[:1])
    with pytest.raises(TypeError, match="flow must have the dtype of the image"):
        warp(image, flow.double())
    with pytest.raises(ValueError, match="to match the image"):
        warp(image, flow[:1])
    with pytest.raises(ValueError, match="mode"):
        warp(image, flow, mode="linear")
    with pytest.raises(ValueError, match="padding_mode"):
        warp(image, flow, padding_mode="reflect")


# ----------------------------------------------------------------------------------------------
# resize_flow
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["bilinear", "nearest", "bicubic", "area"])
@pytest.mark.parametrize("size", [(24, 60), (6, 10), (12, 20), (7, 33)])
def test_resize_flow_constant(device: torch.device, dtype: torch.dtype, mode: str, size: tuple[int, int]) -> None:
    flow = translation_flow(12, 20, 1.5, -2.0, batch=2, device=device, dtype=dtype)
    out = resize_flow(flow, size, mode=mode)
    assert out.shape == (2, 2, *size) and out.dtype == dtype and out.device == flow.device
    expected = translation_flow(*size, 1.5 * size[1] / 20, -2.0 * size[0] / 12, batch=2, device=device, dtype=dtype)
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("size", [(48, 30), (9, 13)])
def test_resize_flow_affine(device: torch.device, size: tuple[int, int]) -> None:
    # bilinear resizing (half-pixel centres) reproduces affine fields away from the borders
    height, width = 16, 20
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float64), torch.arange(width, dtype=torch.float64), indexing="ij"
    )
    flow = torch.stack((0.5 + 0.1 * x - 0.05 * y, -1 + 0.02 * x + 0.2 * y)).unsqueeze(0).to(device)
    out = resize_flow(flow, size)
    h2, w2 = size
    y2, x2 = torch.meshgrid(torch.arange(h2, dtype=torch.float64), torch.arange(w2, dtype=torch.float64), indexing="ij")
    xs, ys = (x2 + 0.5) * width / w2 - 0.5, (y2 + 0.5) * height / h2 - 0.5  # positions in the old grid
    expected = torch.stack(((0.5 + 0.1 * xs - 0.05 * ys) * w2 / width, (-1 + 0.02 * xs + 0.2 * ys) * h2 / height))
    inside = (xs >= 0) & (xs <= width - 1) & (ys >= 0) & (ys <= height - 1)
    torch.testing.assert_close(out[0].cpu()[:, inside], expected[:, inside], rtol=0, atol=1e-12)


def resize_matrix(n_in: int, n_out: int, mode: str) -> np.ndarray:
    """``(n_out, n_in)`` resampling matrix of F.interpolate along one axis, written independently."""
    matrix = np.zeros((n_out, n_in))
    d = np.arange(n_out)
    if mode == "nearest":  # "nearest-exact": the pixel containing the output pixel's centre
        matrix[d, np.minimum((2 * d + 1) * n_in // (2 * n_out), n_in - 1)] = 1
    elif mode == "area":  # adaptive average pooling over the pixels the output pixel overlaps
        for i in d:
            start, stop = i * n_in // n_out, -(-(i + 1) * n_in // n_out)
            matrix[i, start:stop] = 1 / (stop - start)
    else:  # half-pixel centres, edge pixels replicated
        coord = (d + 0.5) * n_in / n_out - 0.5
        base = np.floor(coord)
        if mode == "bilinear":
            taps = [(base, 1 - (coord - base)), (base + 1, coord - base)]
        else:
            taps = [(base + k, keys_cubic(coord - base - k)) for k in (-1, 0, 1, 2)]
        for index, weight in taps:
            np.add.at(matrix, (d, np.clip(index, 0, n_in - 1).astype(int)), weight)
    return matrix


@pytest.mark.parametrize("mode", ["bilinear", "nearest", "bicubic", "area"])
@pytest.mark.parametrize("size", [(40, 31), (5, 7), (39, 34), (13, 6), (4, 51)])
def test_resize_flow_matches_reference(device: torch.device, mode: str, size: tuple[int, int]) -> None:
    # a random field pins the sampling positions of every mode; odd input sizes keep nearest-exact
    # away from ties
    height, width = 13, 17
    flow = 5 * torch.randn(2, 2, height, width, device=device, dtype=torch.float64)
    out = resize_flow(flow, size, mode=mode)
    rows, cols = resize_matrix(height, size[0], mode), resize_matrix(width, size[1], mode)
    expected = np.einsum("yi,bcij,xj->bcyx", rows, flow.cpu().numpy(), cols)
    expected *= np.array([size[1] / width, size[0] / height])[:, None, None]
    torch.testing.assert_close(out.cpu(), torch.from_numpy(expected), rtol=0, atol=1e-12)


@pytest.mark.parametrize("mode", ["bilinear", "nearest", "bicubic", "area"])
def test_resize_flow_gradcheck(device: torch.device, mode: str) -> None:
    flow = torch.randn(1, 2, 5, 6, device=device, dtype=torch.float64, requires_grad=True)
    assert gradcheck(lambda f: resize_flow(f, (9, 4), mode=mode), (flow,))


@pytest.mark.cuda
def test_warp_device_mismatch() -> None:
    with pytest.raises(ValueError, match="flow must be on the device of the image"):
        warp(torch.zeros(1, 1, 4, 5), torch.zeros(1, 2, 4, 5, device="cuda"))


def test_resize_flow_validation() -> None:
    flow = torch.zeros(1, 2, 8, 9)
    with pytest.raises(ValueError, match="flow"):
        resize_flow(torch.zeros(1, 3, 8, 9), (4, 4))
    with pytest.raises(ValueError, match="size"):
        resize_flow(flow, (0, 4))
    with pytest.raises(ValueError, match="mode"):
        resize_flow(flow, (4, 4), mode="linear")

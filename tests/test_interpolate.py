"""Tests of interpolate_frames."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
import torch
from synthetic import AnalyticTexture, flow_to_cv, image_to_cv, make_pair, rotation_flow, translation_flow
from torch import Tensor
from torch.autograd import gradcheck

from opticalflow_torch import interpolate_frames, warp
from opticalflow_torch.interpolate import _SMOOTHING, _match_distribution


def rain(image: Tensor) -> Tensor:
    """Radar-like rain rates from a 0-255 texture: about half dry (exact zeros), peaks near 30."""
    return torch.relu((image - 140) / 4)


def random_inputs(
    batch: int = 2,
    channels: int = 2,
    height: int = 12,
    width: int = 14,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Frames without ties in [0, 10], a flow and a backward flow with fractional displacements."""
    frame0 = torch.rand(batch, channels, height, width, device=device, dtype=dtype) * 10
    frame1 = torch.rand(batch, channels, height, width, device=device, dtype=dtype) * 10
    flow = torch.randn(batch, 2, height, width, device=device, dtype=dtype) * 1.5
    backward_flow = torch.randn(batch, 2, height, width, device=device, dtype=dtype) * 1.5
    return frame0, frame1, flow, backward_flow


def formula(
    frame0: Tensor, frame1: Tensor, flow: Tensor, times: Sequence[float], backward_flow: Tensor | None = None
) -> Tensor:
    """The documented blend, with warp, one time after the other."""
    out = []
    for t in times:
        motion1 = (1 - t) * flow if backward_flow is None else -(1 - t) * backward_flow
        out.append((1 - t) * warp(frame0, -t * flow) + t * warp(frame1, motion1))
    return torch.stack(out, dim=1)


# ----------------------------------------------------------------------------------------------
# Blend
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("match", [False, True])
@pytest.mark.parametrize(
    ("t", "steps"), [(0.4, 1), ([0.2, 0.5, 0.9], 3), ((0.5,), 1), ("tensor", 4), ("scalar tensor", 1)]
)
def test_shape_dtype_device(device: torch.device, dtype: torch.dtype, match: bool, t: object, steps: int) -> None:
    frame0, frame1, flow, _ = random_inputs(device=device, dtype=dtype)
    if t == "tensor":
        t = torch.tensor([0.0, 0.25, 0.5, 1.0], dtype=torch.float64)  # cast to the frames' dtype and device
    elif t == "scalar tensor":
        t = torch.tensor(0.3, device=device)
    out = interpolate_frames(frame0, frame1, flow, t, match_distribution=match)
    assert out.shape == (2, steps, 2, 12, 14)
    assert out.dtype == dtype and out.device == frame0.device
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("backward", [False, True])
def test_matches_formula(device: torch.device, dtype: torch.dtype, backward: bool) -> None:
    frame0, frame1, flow, backward_flow = random_inputs(device=device, dtype=dtype)
    backward_flow = backward_flow if backward else None
    times = [0.1, 0.5, 0.8]
    out = interpolate_frames(frame0, frame1, flow, times, backward_flow=backward_flow)
    expected = formula(frame0, frame1, flow, times, backward_flow)
    torch.testing.assert_close(out, expected, rtol=0, atol=100 * torch.finfo(dtype).eps)


@pytest.mark.parametrize("match", [False, True])
@pytest.mark.parametrize("backward", [False, True])
def test_end_times_are_the_frames(device: torch.device, dtype: torch.dtype, match: bool, backward: bool) -> None:
    frame0, frame1, flow, backward_flow = random_inputs(device=device, dtype=dtype)
    t = torch.tensor([0.0, 1.0], device=device, dtype=dtype)
    backward_flow = backward_flow if backward else None
    out = interpolate_frames(frame0, frame1, flow, t, backward_flow=backward_flow, match_distribution=match)
    assert torch.equal(out[:, 0], frame0)
    assert torch.equal(out[:, 1], frame1)


@pytest.mark.parametrize("match", [False, True])
@pytest.mark.parametrize("backward", [False, True])
def test_end_time_gradients_are_the_blend_ones(device: torch.device, match: bool, backward: bool) -> None:
    """All gradients at t = 0 and 1, including the one with respect to t, are those of the formula."""
    frame0, frame1, flow, backward_flow = random_inputs(device=device)
    t = torch.tensor([0.0, 1.0], device=device, dtype=torch.float64)
    inputs = [x.requires_grad_() for x in (frame0, frame1, flow, t, backward_flow)]
    backward_flow = backward_flow if backward else None
    out = interpolate_frames(
        frame0, frame1, flow, t, backward_flow=backward_flow, match_distribution=match, quantiles=32
    )
    expected = formula(frame0, frame1, flow, t, backward_flow)
    if match:
        expected = _match_distribution(expected, frame0, frame1, t, 32)
    weights = torch.rand_like(out)
    grads = torch.autograd.grad((out * weights).sum(), inputs, allow_unused=True)
    grads_expected = torch.autograd.grad((expected * weights).sum(), inputs, allow_unused=True)
    for grad, grad_expected in zip(grads, grads_expected, strict=True):
        torch.testing.assert_close(grad, grad_expected)
    assert (grads[3].abs() > 1).all()  # the frames differ, so the blend changes with t


@pytest.mark.parametrize("t", [0.25, 0.5, 0.75])
def test_translation(t: float) -> None:
    """Both warps land on the texture moved by (1 - t) * flow; a cross-fade does not."""
    height, width = 48, 56
    texture = AnalyticTexture(1, min_wavelength=16)
    flow = translation_flow(height, width, 7.3, -5.4, dtype=torch.float64)
    frame0, frame1 = make_pair(texture, flow)
    truth = texture.image(height, width, (1 - t) * flow, dtype=torch.float64)
    inner = (..., slice(8, -8), slice(8, -8))  # away from the replicated borders
    for backward_flow in (None, -flow):
        out = interpolate_frames(frame0, frame1, flow, t, backward_flow=backward_flow)[:, 0]
        error = (out - truth)[inner].abs()
        crossfade = ((1 - t) * frame0 + t * frame1 - truth)[inner].abs()
        assert error.mean() < 0.06 * crossfade.mean()  # measured 0.033-0.038
        assert error.max() < 1.5  # bilinear interpolation of 0-255 waves of 16 pixels and more


def test_batch_and_channel_independence(device: torch.device) -> None:
    frame0, frame1, flow, backward_flow = random_inputs(batch=3, channels=2, device=device)
    for match in (False, True):
        out = interpolate_frames(
            frame0, frame1, flow, [0.3, 0.6], backward_flow=backward_flow, match_distribution=match
        )
        for b in range(3):
            for c in range(2):
                single = interpolate_frames(
                    frame0[b : b + 1, c : c + 1],
                    frame1[b : b + 1, c : c + 1],
                    flow[b : b + 1],
                    [0.3, 0.6],
                    backward_flow=backward_flow[b : b + 1],
                    match_distribution=match,
                )
                torch.testing.assert_close(out[b : b + 1, :, c : c + 1], single, rtol=0, atol=1e-12)


def test_time_forms_agree(device: torch.device) -> None:
    frame0, frame1, flow, _ = random_inputs(device=device)
    expected = interpolate_frames(frame0, frame1, flow, [0.25, 0.5])
    forms = [
        (0.25, 0.5),
        torch.tensor([0.25, 0.5]),
        [torch.tensor(0.25), 0.5],
        [torch.tensor([0.25], device=device), np.float64(0.5)],
    ]
    for t in forms:
        torch.testing.assert_close(interpolate_frames(frame0, frame1, flow, t), expected, rtol=0, atol=1e-12)
    torch.testing.assert_close(interpolate_frames(frame0, frame1, flow, 0.5)[:, 0], expected[:, 1])


@pytest.mark.cuda
@pytest.mark.parametrize("match", [False, True])
def test_cpu_cuda_agree(match: bool) -> None:
    height, width = 40, 48
    flow = rotation_flow(height, width, 5.0) + translation_flow(height, width, 2.2, -1.3)
    frame0, frame1 = (rain(frame) for frame in make_pair(AnalyticTexture(2), flow))
    t = [0.2, 0.5, 0.9]
    out = interpolate_frames(frame0.cuda(), frame1.cuda(), flow.cuda(), t, match_distribution=match).cpu()
    ref = interpolate_frames(frame0, frame1, flow, t, match_distribution=match)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-4)  # values up to 30


# ----------------------------------------------------------------------------------------------
# Probability matching
# ----------------------------------------------------------------------------------------------


def bilinear(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """scipy's ``map_coordinates(image, (y, x), order=1, mode="nearest")``, exactly, in float64."""
    height, width = image.shape
    x, y = np.clip(x, 0, width - 1), np.clip(y, 0, height - 1)
    x0, y0 = np.minimum(np.floor(x).astype(int), width - 2), np.minimum(np.floor(y).astype(int), height - 2)
    fx, fy = x - x0, y - y0
    image = image.astype(np.float64)
    top = (1 - fx) * image[y0, x0] + fx * image[y0, x0 + 1]
    bottom = (1 - fx) * image[y0 + 1, x0] + fx * image[y0 + 1, x0 + 1]
    return (1 - fy) * top + fy * bottom


def livemap_interpolate(
    a: np.ndarray, b: np.ndarray, flow: np.ndarray | None, f: float, blend: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """meteotn-bot's livemap advection correction and probability matching for one gap and a given flow.

    A given ``blend`` skips the warping: only the rank matching is applied to it.
    """
    if blend is None:
        height, width = a.shape
        grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
        u, v = flow[..., 0].astype(np.float64), flow[..., 1].astype(np.float64)
        warp_a = bilinear(a, grid_x - f * u, grid_y - f * v)
        warp_b = bilinear(b, grid_x + (1 - f) * u, grid_y + (1 - f) * v)
        blend = (1 - f) * warp_a + f * warp_b
    target = (1 - f) * np.sort(a.ravel()) + f * np.sort(b.ravel())
    order = np.argsort(blend, axis=None, kind="stable")
    inter = np.empty(blend.size)
    inter[order] = target
    return inter.reshape(blend.shape), blend


@pytest.mark.parametrize("t", [0.25, 0.5, 0.75])
def test_matching_against_livemap(t: float) -> None:
    """With one quantile per pixel the matching is close to livemap's rank matching."""
    height, width = 96, 128
    flow = rotation_flow(height, width, 4.0) + translation_flow(height, width, 2.5, -1.5)
    frame0, frame1 = (rain(frame) for frame in make_pair(AnalyticTexture(3), flow))
    a, b = image_to_cv(frame0), image_to_cv(frame1)
    reference, reference_blend = livemap_interpolate(a, b, flow_to_cv(flow), t)
    blend = interpolate_frames(frame0, frame1, flow, t)[0, 0, 0].numpy()
    ranked, _ = livemap_interpolate(a, b, None, t, blend)  # the rank matching alone, on the same blend
    close = interpolate_frames(frame0, frame1, flow, t, match_distribution=True, quantiles=height * width)
    same_blend = np.abs(close[0, 0, 0].numpy() - ranked)
    approx = interpolate_frames(frame0, frame1, flow, t, match_distribution=True)[0, 0, 0].numpy()
    diff256 = np.abs(approx - reference)  # the whole pipeline
    print(
        f"\nt={t}: values up to {reference.max():.1f}, {np.mean(reference == 0):.0%} dry; blend vs reference max "
        f"{np.abs(blend - reference_blend).max():.2g}; H*W quantiles vs rank matching of the same blend: mean "
        f"{same_blend.mean():.2g}, max {same_blend.max():.2g}; 256 quantiles, whole pipeline: mean "
        f"{diff256.mean():.2g}, p99 {np.quantile(diff256, 0.99):.2g}, max {diff256.max():.2g}"
    )
    assert np.abs(blend - reference_blend).max() < 1e-4  # float32; measured up to 5.5e-5
    assert same_blend.mean() < 1e-3 and same_blend.max() < 0.05  # measured up to 2.6e-4 and 0.013
    assert diff256.mean() < 0.01 and diff256.max() < 0.4  # measured up to 0.0062 and 0.19


def test_matching_reproduces_target_quantiles(device: torch.device, dtype: torch.dtype) -> None:
    frame0, frame1, flow, _ = random_inputs(height=20, width=24, device=device, dtype=dtype)
    t = torch.tensor([0.3, 0.8], device=device, dtype=dtype).view(1, 2, 1, 1)
    target = (1 - t) * frame0.flatten(-2).sort().values.unsqueeze(1) + t * frame1.flatten(-2).sort().values.unsqueeze(1)
    # one quantile per pixel: the targets, up to the smoothing of the blend's distribution function
    out = interpolate_frames(frame0, frame1, flow, t.flatten(), match_distribution=True, quantiles=20 * 24)
    error = (out.flatten(-2).sort().values - target).abs()
    assert error.max() < 0.1 and error.mean() < 0.01  # values in [0, 10]; measured up to 0.07 and 0.0046
    assert ((out.flatten(-2).mean(-1) - target.mean(-1)).abs() < 0.01).all()  # measured up to 0.003
    # fewer quantiles: also up to the piecewise-linear approximation
    out = interpolate_frames(frame0, frame1, flow, t.flatten(), match_distribution=True, quantiles=64)
    probabilities = torch.linspace(0, 1, 11, device=device, dtype=dtype)
    got = torch.quantile(out.flatten(-2), probabilities, dim=-1)
    want = torch.quantile(target, probabilities, dim=-1)
    torch.testing.assert_close(got, want, rtol=0, atol=0.15)  # measured up to 0.09
    torch.testing.assert_close(got[0], want[0])  # the minimum is exact
    assert (got[-1] <= want[-1]).all()


def test_matching_keeps_ranks(device: torch.device) -> None:
    """The map is non-decreasing: the order of the blend's values is kept."""
    frame0, frame1, flow, _ = random_inputs(height=20, width=24, device=device)
    blend = interpolate_frames(frame0, frame1, flow, [0.4]).flatten(-2)
    matched = interpolate_frames(frame0, frame1, flow, [0.4], match_distribution=True, quantiles=32).flatten(-2)
    assert (matched.gather(-1, blend.argsort(dim=-1)).diff(dim=-1) >= 0).all()


@pytest.mark.parametrize("quantiles", [16, 120, 256])
def test_matching_ties(device: torch.device, quantiles: int) -> None:
    """Ties stay tied, between the targets of their ranks; the tie at the blend's minimum takes the lowest target."""
    t = 0.25
    height, width = 10, 12
    frame0, frame1, _, _ = random_inputs(batch=1, channels=1, height=height, width=width, device=device)
    frame0.requires_grad_()
    frame1.requires_grad_()
    target = (1 - t) * frame0.flatten().sort().values + t * frame1.flatten().sort().values
    # flow along x only: each warp reads one border column, so every row of the blend is one tie
    flow = torch.zeros(1, 2, height, width, device=device, dtype=torch.float64)
    flow[:, 0] = 100.0
    flow.requires_grad_()
    blend = interpolate_frames(frame0, frame1, flow, t)
    assert (blend == blend[..., :1]).all() and blend[..., 0].unique().numel() == height
    out = interpolate_frames(frame0, frame1, flow, t, match_distribution=True, quantiles=quantiles)
    assert (out == out[..., :1]).all()  # ties stay tied
    rows = blend[0, 0, 0, :, 0].argsort()
    values, ranks = out[0, 0, 0, rows, 0], target.view(height, width)  # row r holds ranks r W to (r + 1) W - 1
    assert values[0] == target[0]
    assert ((values >= ranks[:, 0]) & (values <= ranks[:, -1])).all()
    grads = torch.autograd.grad(out.square().sum(), (frame0, frame1, flow))
    assert all(torch.isfinite(g).all() for g in grads)
    # both warps read one corner pixel everywhere: the whole blend is the tie at the minimum
    flow = torch.full_like(flow, 100.0, requires_grad=True)
    out = interpolate_frames(frame0, frame1, flow, t, match_distribution=True, quantiles=quantiles)
    torch.testing.assert_close(out, target[0].expand_as(out))
    grads = torch.autograd.grad(out.square().sum(), (frame0, frame1, flow))
    assert all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("quantiles", [17, 64, 100, 256])
def test_matching_is_continuous_at_ties(device: torch.device, quantiles: int) -> None:
    """Separating values tied in the blend, at its minimum or above, moves the matched values continuously."""
    frame0, frame1 = (255 * torch.rand(1, 1, 10, 10, device=device, dtype=torch.float64) for _ in range(2))
    for frame in (frame0, frame1):
        frame[..., 0, 1] = frame[..., 0, 0]  # a tie in the blend, between rain values
        frame[..., 1, :] = 0.0  # and 10% dry
    flow = torch.zeros(1, 2, 10, 10, device=device, dtype=torch.float64)
    blend = interpolate_frames(frame0, frame1, flow, 0.5)
    drier = torch.where(torch.arange(100, device=device).view(10, 10) >= 70, 0.0, blend)  # 40% dry
    t = torch.tensor([0.5], device=device, dtype=torch.float64)
    base = _match_distribution(drier, frame0, frame1, t, quantiles)
    assert (base[drier == 0] == 0).all() and base.flatten()[0] == base.flatten()[1]
    for index in (1, 15, 75):  # in the rain tie, dry in both frames, dry in the blend only
        for delta in (1e-3, 1e-6, 1e-9):
            nudged = drier.clone()
            nudged.view(-1)[index] += delta
            change = (_match_distribution(nudged, frame0, frame1, t, quantiles) - base).abs().max()
            assert change < 50 * delta + 1e-11  # measured up to 15 delta; 1e11 delta before the smoothing
    # the same through interpolate_frames, nudging a frame
    nudged = frame0.clone()
    nudged[..., 0, 1] += 1e-9
    out = interpolate_frames(nudged, frame1, flow, 0.5, match_distribution=True, quantiles=quantiles)
    assert (
        out - interpolate_frames(frame0, frame1, flow, 0.5, match_distribution=True, quantiles=quantiles)
    ).abs().max() < 1e-8


@pytest.mark.parametrize("quantiles", [16, 256, 128 * 128])
def test_matching_keeps_the_dry_area_of_a_drier_blend(device: torch.device, dtype: torch.dtype, quantiles: int) -> None:
    """Rain entering through the border is missing from both warps, so the blend is drier than the target.

    Its dry pixels must stay exactly dry, not all take the same drizzle, and its rain must stay rain above the
    smoothing of the distribution function.
    """
    size = 128
    coords = torch.arange(size, device=device, dtype=dtype)
    y, x = torch.meshgrid(coords, coords, indexing="ij")

    def cell(cx: float, cy: float, sigma: float, peak: float) -> Tensor:
        rate = peak * torch.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))
        return torch.where(rate > 0.1, rate, 0.0)

    frame0 = (cell(50, 64, 6, 20) + cell(-14, 30, 8, 30)).view(1, 1, size, size)
    frame1 = (cell(62, 64, 6, 20) + cell(-2, 30, 8, 30)).view(1, 1, size, size)
    flow = torch.zeros(1, 2, size, size, device=device, dtype=dtype)
    flow[:, 0] = 12.0
    t = torch.tensor([0.25, 0.5, 0.75], device=device, dtype=dtype)
    blend = interpolate_frames(frame0, frame1, flow, t)
    target = (1 - t.view(-1, 1)) * frame0.flatten().sort().values + t.view(-1, 1) * frame1.flatten().sort().values
    assert ((blend == 0).flatten(2).double().mean(-1) > (target == 0).double().mean(-1) + 0.005).all()
    frame0.requires_grad_()
    out = interpolate_frames(frame0, frame1, flow, t, match_distribution=True, quantiles=quantiles)
    assert (out[blend == 0] == 0).all()
    half = _SMOOTHING * (target[:, -1] - target[:, 0]) / (quantiles - 1)  # 0.41 at 16 quantiles, 0.024 at 256
    assert (out[blend > half.view(1, -1, 1, 1, 1)] > 0).all()
    (grad,) = torch.autograd.grad(out.sum(), frame0)
    assert torch.isfinite(grad).all()


@pytest.mark.parametrize("quantiles", [256, 32 * 40])
def test_dry_frames(device: torch.device, dtype: torch.dtype, quantiles: int) -> None:
    """Mostly zeros (ties) and all zeros: finite values and gradients, and dry stays dry."""
    height, width = 32, 40
    flow = rotation_flow(height, width, 6.0) + translation_flow(height, width, 1.7, 0.6)
    frame0, frame1 = (rain(frame) for frame in make_pair(AnalyticTexture(5), flow, device=device, dtype=dtype))
    flow = flow.to(device=device, dtype=dtype)
    assert (frame0 == 0).float().mean() > 0.4
    for f0, f1 in ((frame0, frame1), (torch.zeros_like(frame0), torch.zeros_like(frame1))):
        f0, f1, fl, backward = (x.clone().requires_grad_() for x in (f0, f1, flow, -flow))
        t = torch.tensor([0.0, 0.3, 0.6, 1.0], device=device, dtype=dtype, requires_grad=True)
        for backward_flow in (None, backward):
            kwargs = {"backward_flow": backward_flow, "quantiles": quantiles}
            out = interpolate_frames(f0, f1, fl, t, match_distribution=True, **kwargs)
            blend = interpolate_frames(f0, f1, fl, t, **kwargs)
            assert torch.isfinite(out).all()
            assert (out[blend == 0] == 0).all()
            inputs = (f0, f1, fl, t) if backward_flow is None else (f0, f1, fl, t, backward)
            grads = torch.autograd.grad((out * torch.rand_like(out)).sum(), inputs)
            assert all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
@pytest.mark.parametrize("where", ["frame1", "flow"])
def test_matching_non_finite(device: torch.device, where: str, value: float) -> None:
    # A NaN position in the target's quantile function used to become the index -2**63: an
    # IndexError on the CPU, and on CUDA a device-side assert that ends the process.
    frame0, frame1, flow, _ = random_inputs(batch=2, height=16, width=18, device=device)
    expected = interpolate_frames(frame0[1:], frame1[1:], flow[1:], [0.3, 0.6], match_distribution=True)
    (frame1 if where == "frame1" else flow)[0, 0, 7, 9] = value
    out = interpolate_frames(frame0, frame1, flow, [0.3, 0.6], match_distribution=True)
    assert not out[0].isfinite().all()
    torch.testing.assert_close(out[1:], expected, rtol=0, atol=1e-12)


# ----------------------------------------------------------------------------------------------
# Gradients and compilation
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("match", [False, True])
@pytest.mark.parametrize("backward", [False, True])
def test_gradcheck(device: torch.device, match: bool, backward: bool) -> None:
    frame0, frame1, flow, backward_flow = random_inputs(channels=1, height=5, width=6, device=device)
    t = torch.tensor([0.35, 0.7], device=device, dtype=torch.float64)
    inputs = [x.requires_grad_() for x in (frame0, frame1, flow, t, backward_flow)]

    def fn(f0: Tensor, f1: Tensor, fl: Tensor, tt: Tensor, bf: Tensor) -> Tensor:
        bf = bf if backward else None
        return interpolate_frames(f0, f1, fl, tt, backward_flow=bf, match_distribution=match, quantiles=8)

    nondet_tol = 1e-12 if device.type == "cuda" else 0.0  # grid_sample's CUDA backward adds atomically
    assert gradcheck(fn, inputs, nondet_tol=nondet_tol)


def test_times_in_a_list_get_gradients(device: torch.device) -> None:
    frame0, frame1, flow, _ = random_inputs(device=device)
    time = torch.tensor(0.4, device=device, dtype=torch.float64, requires_grad=True)
    for match in (False, True):
        out = interpolate_frames(frame0, frame1, flow, [0.2, time], match_distribution=match)
        (grad,) = torch.autograd.grad(out.square().sum(), time)
        assert torch.isfinite(grad) and grad != 0


@pytest.mark.compile
@pytest.mark.parametrize("match", [False, True])
def test_compile(device: torch.device, match: bool) -> None:
    torch._dynamo.reset()
    frame0, frame1, flow, backward_flow = random_inputs(height=20, width=24, device=device, dtype=torch.float32)
    t = torch.tensor([0.0, 0.3, 0.7], device=device, requires_grad=True)
    compiled = torch.compile(interpolate_frames, fullgraph=True)
    kwargs = {"backward_flow": backward_flow, "match_distribution": match, "quantiles": 64}
    expected = interpolate_frames(frame0, frame1, flow, t, **kwargs)
    out = compiled(frame0, frame1, flow, t, **kwargs)
    # values up to 10 in float32; compiled kernels round differently, by up to 7e-5 after the matching (torch 2.7)
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=2e-4)
    weights = torch.rand_like(out)
    (grad,) = torch.autograd.grad((out * weights).sum(), t)
    (grad_expected,) = torch.autograd.grad((expected * weights).sum(), t)
    torch.testing.assert_close(grad[1:], grad_expected[1:], rtol=1e-4, atol=1e-3)  # t = 0 sits on a bilinear kink


# ----------------------------------------------------------------------------------------------
# Arguments
# ----------------------------------------------------------------------------------------------


def test_argument_errors() -> None:
    frame0, frame1, flow, backward_flow = random_inputs()
    with pytest.raises(ValueError, match=r"t must be in \[0, 1\]"):
        interpolate_frames(frame0, frame1, flow, 1.5)
    with pytest.raises(ValueError, match=r"t must be in \[0, 1\]"):
        interpolate_frames(frame0, frame1, flow, [0.5, -0.1])
    with pytest.raises(ValueError, match="at least one time"):
        interpolate_frames(frame0, frame1, flow, [])
    with pytest.raises(ValueError, match="1-D"):
        interpolate_frames(frame0, frame1, flow, torch.zeros(2, 2))
    with pytest.raises(TypeError, match="t must be"):
        interpolate_frames(frame0, frame1, flow, "0.5")
    with pytest.raises(TypeError, match="t must be"):
        interpolate_frames(frame0, frame1, flow, [0.5, "x"])
    with pytest.raises(ValueError, match="scalar"):
        interpolate_frames(frame0, frame1, flow, [torch.zeros(2)])
    for quantiles in (1, 2.5, True):
        with pytest.raises(ValueError, match="quantiles"):
            interpolate_frames(frame0, frame1, flow, 0.5, match_distribution=True, quantiles=quantiles)
    with pytest.raises(ValueError, match="same shape"):
        interpolate_frames(frame0, frame1[:, :1], flow, 0.5)
    with pytest.raises(TypeError, match="same dtype"):
        interpolate_frames(frame0, frame1.float(), flow, 0.5)
    with pytest.raises(ValueError, match="flow must have shape"):
        interpolate_frames(frame0, frame1, flow[:, :, :-1], 0.5)
    with pytest.raises(ValueError, match="backward_flow must have shape"):
        interpolate_frames(frame0, frame1, flow, 0.5, backward_flow=backward_flow[:1])
    with pytest.raises(TypeError, match="backward_flow must have the dtype"):
        interpolate_frames(frame0, frame1, flow, 0.5, backward_flow=backward_flow.float())

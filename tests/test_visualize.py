"""Tests of flow_to_rgb, against a NumPy port of the Middlebury colour coding with the wheel closed at +x."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch import Tensor
from torch.autograd import gradcheck, gradgradcheck

from opticalflow_torch import flow_to_rgb


def middlebury_wheel() -> np.ndarray:
    """The colour wheel of Baker et al.'s ``computeColor``, shape (55, 3), 0-255, as in flow_vis and torchvision."""
    ry, yg, gc, cb, bm, mr = 15, 6, 4, 11, 13, 6
    wheel = np.zeros((ry + yg + gc + cb + bm + mr, 3))
    col = 0
    wheel[col : col + ry, 0] = 255
    wheel[col : col + ry, 1] = np.floor(255 * np.arange(ry) / ry)
    col += ry
    wheel[col : col + yg, 0] = 255 - np.floor(255 * np.arange(yg) / yg)
    wheel[col : col + yg, 1] = 255
    col += yg
    wheel[col : col + gc, 1] = 255
    wheel[col : col + gc, 2] = np.floor(255 * np.arange(gc) / gc)
    col += gc
    wheel[col : col + cb, 1] = 255 - np.floor(255 * np.arange(cb) / cb)
    wheel[col : col + cb, 2] = 255
    col += cb
    wheel[col : col + bm, 2] = 255
    wheel[col : col + bm, 0] = np.floor(255 * np.arange(bm) / bm)
    col += bm
    wheel[col : col + mr, 2] = 255 - np.floor(255 * np.arange(mr) / mr)
    wheel[col : col + mr, 0] = 255
    return wheel


def reference_rgb(flow: Tensor, max_magnitude: float | None = None) -> Tensor:
    """flow_vis' ``flow_uv_to_colors`` in float64 without the 8-bit rounding, per image, with the 55 colours over 55
    steps of the circle instead of 54 (closing the wheel); beyond the maximum magnitude the colour is divided by the
    relative magnitude, as in flowpy."""
    wheel = middlebury_wheel()
    count = wheel.shape[0]
    out = []
    for image in flow.detach().double().cpu().numpy():
        u, v = image
        scale = np.sqrt(u**2 + v**2).max() if max_magnitude is None else max_magnitude
        u, v = u / scale, v / scale
        radius = np.sqrt(u**2 + v**2)
        position = (np.arctan2(-v, -u) / np.pi + 1) / 2 * count
        k0 = np.minimum(np.floor(position).astype(int), count - 1)
        k1 = k0 + 1
        k1[k1 == count] = 0
        f = position - k0
        rgb = []
        for i in range(3):
            colour = (1 - f) * wheel[k0, i] / 255 + f * wheel[k1, i] / 255
            rgb.append(np.where(radius <= 1, 1 - radius * (1 - colour), colour / np.maximum(radius, 1)))
        out.append(np.stack(rgb))
    return torch.from_numpy(np.stack(out))


def direction_flow(angles: Tensor, magnitudes: Tensor | float = 1.0) -> Tensor:
    """A one-row flow, one pixel per angle (radians, from +x towards +y), shape (1, 2, 1, N)."""
    return (magnitudes * torch.stack((torch.cos(angles), torch.sin(angles)))).view(1, 2, 1, -1)


def directional_derivatives(flow: Tensor, direction: Tensor, step: float, **kwargs: object) -> tuple[Tensor, Tensor]:
    """Autograd's derivative of the colour of a one-pixel ``flow`` along ``direction``, and the one-sided finite
    difference with ``step`` (negative for the other side)."""
    flow = flow.reshape(1, 2, 1, 1).clone().requires_grad_()
    rgb = flow_to_rgb(flow, **kwargs).flatten()
    grads = [torch.autograd.grad(channel, flow, retain_graph=True)[0].flatten() @ direction for channel in rgb]
    moved = flow_to_rgb(flow.detach() + step * direction.view(1, 2, 1, 1), **kwargs).flatten()
    return torch.stack(grads), (moved - rgb.detach()) / step


def test_matches_reference(device: torch.device, dtype: torch.dtype) -> None:
    flow = 3 * torch.randn(2, 2, 20, 30, dtype=torch.float64)
    flow[1] *= 0.01  # per-image maximum
    out = flow_to_rgb(flow.to(device=device, dtype=dtype))
    assert out.shape == (2, 3, 20, 30) and out.dtype == dtype and out.device.type == device.type
    assert out.min() >= 0 and out.max() <= 1
    # an angle off by eps moves the wheel position by 54 / (2 pi) eps, a colour by up to 255/4 / 255 per position
    tolerance = 100 * torch.finfo(dtype).eps
    torch.testing.assert_close(out.cpu().double(), reference_rgb(flow), rtol=0, atol=tolerance)


def test_wheel_colours(device: torch.device) -> None:
    """At the wheel's own angles and full magnitude, the 55 colours of the Middlebury wheel."""
    k = torch.arange(55, dtype=torch.float64)
    # position k is atan2(-v, -u) = pi (2 k / 55 - 1), i.e. the flow points at angle 2 pi k / 55
    out = flow_to_rgb(direction_flow(2 * math.pi * k / 55).to(device))
    expected = torch.from_numpy(middlebury_wheel()).T / 255
    torch.testing.assert_close(out[0, :, 0].cpu(), expected, rtol=0, atol=1e-12)


def test_known_directions(device: torch.device, dtype: torch.dtype) -> None:
    """Right is red, down (+y, rows) orange-yellow, left cyan-blue and up blue-violet; half magnitude is paler."""
    angles = torch.tensor([0.0, 0.5, 1.0, 1.5, 0.0], dtype=torch.float64) * math.pi
    magnitudes = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.5], dtype=torch.float64)
    out = flow_to_rgb(direction_flow(angles, magnitudes).to(device=device, dtype=dtype))[0, :, 0].T
    # positions 0, 13.75, 27.5 and 41.25 of the wheel
    expected = torch.tensor(
        [[255, 0, 0], [255, 233.75, 0], [0, 197.5, 255], [102.75, 0, 255], [255, 127.5, 127.5]],
        dtype=dtype,
        device=device,
    )
    torch.testing.assert_close(out, expected / 255, rtol=0, atol=1e-6)


def test_continuous_around_the_circle(device: torch.device) -> None:
    """No jump at +x, where the wheel closes: equal flows (v = 0.0 or -0.0) get equal colours."""
    count = 200_000
    angles = torch.linspace(-math.pi, math.pi, count + 1, dtype=torch.float64)
    out = flow_to_rgb(direction_flow(angles).to(device), max_magnitude=1.0)[0, :, 0].cpu()
    # at most 55 / (2 pi) wheel positions per radian, and a colour moves by at most 255 / 4 / 255 per position
    bound = 55 / (2 * math.pi) * 0.25 * (2 * math.pi / count)
    assert out.diff(dim=1).abs().max() < 1.01 * bound
    flows = torch.tensor([[1.0, 1.0, 1.0, 1.0], [0.0, -0.0, 1e-9, -1e-9]], dtype=torch.float64, device=device)
    out = flow_to_rgb(flows.view(1, 2, 1, 4), max_magnitude=1.0)[0, :, 0].T.cpu()
    torch.testing.assert_close(out, torch.tensor([[1.0, 0.0, 0.0]]).double().expand(4, 3), rtol=0, atol=1e-8)
    # a kink there: the gradient is the one-sided derivative on the side of the zero's sign
    up = torch.tensor([0.0, 1.0], dtype=torch.float64, device=device)
    for sign in (1.0, -1.0):
        flow = torch.tensor([1.0, sign * 0.0], dtype=torch.float64, device=device)
        grad, one_sided = directional_derivatives(flow, up, sign * 1e-7, max_magnitude=1.0)
        torch.testing.assert_close(grad, one_sided, rtol=0, atol=1e-6)


def test_zero_flow_is_white(device: torch.device, dtype: torch.dtype) -> None:
    flow = torch.zeros(2, 2, 4, 5, device=device, dtype=dtype, requires_grad=True)
    out = flow_to_rgb(flow)
    assert torch.equal(out, torch.ones_like(out))
    (grad,) = torch.autograd.grad(out.sum(), flow)
    assert torch.isfinite(grad).all()
    # still pixels in a moving flow
    moving = torch.randn(1, 2, 4, 5, device=device, dtype=dtype)
    moving[..., 1, :] = 0
    moving.requires_grad_()
    out = flow_to_rgb(moving)
    assert torch.equal(out[..., 1, :], torch.ones_like(out[..., 1, :]))
    (grad,) = torch.autograd.grad(out.sum(), moving)
    assert torch.isfinite(grad).all()


def test_max_magnitude(device: torch.device) -> None:
    flow = 2 * torch.randn(2, 2, 6, 7, dtype=torch.float64)
    torch.testing.assert_close(flow_to_rgb(10 * flow.to(device)), flow_to_rgb(flow.to(device)))  # per image
    for max_magnitude in (1.0, 2.5, 100.0):
        out = flow_to_rgb(flow.to(device), max_magnitude=max_magnitude).cpu()
        torch.testing.assert_close(out, reference_rgb(flow, max_magnitude), rtol=0, atol=1e-12)
    # beyond the maximum the colour darkens continuously
    angle = torch.full((5,), 0.3, dtype=torch.float64)
    magnitudes = torch.tensor([1.0, 1.0 + 1e-9, 2.0, 4.0, 1e6], dtype=torch.float64)
    out = flow_to_rgb(direction_flow(angle, magnitudes).to(device), max_magnitude=1.0)[0, :, 0].cpu()
    torch.testing.assert_close(out[:, 1], out[:, 0], rtol=0, atol=1e-8)
    torch.testing.assert_close(out[:, 2:], out[:, :1] / magnitudes[2:])
    tensor_max = torch.tensor(2.0, device=device, dtype=torch.float64)
    torch.testing.assert_close(
        flow_to_rgb(flow.to(device), max_magnitude=tensor_max), flow_to_rgb(flow.to(device), max_magnitude=2.0)
    )


@pytest.mark.parametrize("max_magnitude", [None, 1.5])
def test_non_finite_pixels(device: torch.device, max_magnitude: float | None) -> None:
    """NaN colours with zero gradient; the rest, gradients included, as if those pixels did not move."""
    flow = torch.randn(1, 2, 6, 7, device=device, dtype=torch.float64)
    clean = flow.clone()
    clean[..., 2, 3] = clean[..., 4, 1] = 0  # does not change the maximum
    flow[..., 2, 3] = float("nan")
    flow[0, 0, 4, 1] = float("inf")
    flow.requires_grad_()
    clean.requires_grad_()
    out, expected = flow_to_rgb(flow, max_magnitude=max_magnitude), flow_to_rgb(clean, max_magnitude=max_magnitude)
    bad = torch.zeros(6, 7, dtype=torch.bool, device=device)
    bad[2, 3] = bad[4, 1] = True
    assert out[..., bad].isnan().all()
    torch.testing.assert_close(out[..., ~bad], expected[..., ~bad])
    weights = torch.rand_like(out)
    (grad,) = torch.autograd.grad(torch.where(bad, 0, out * weights).sum(), flow)
    (grad_expected,) = torch.autograd.grad((expected * weights).sum(), clean)  # zero at still pixels
    torch.testing.assert_close(grad, grad_expected)


def test_gradient_at_max_magnitude(device: torch.device) -> None:
    """Exactly at max_magnitude the gradient is the one-sided derivative from inside, not 0."""
    for vector, max_magnitude in (((0.0, 2.0), 2.0), ((2.0, 0.0), 2.0), ((3.0, 4.0), 5.0)):
        flow = torch.tensor(vector, dtype=torch.float64, device=device)
        grad, inside = directional_derivatives(flow, flow / flow.norm(), -1e-7, max_magnitude=max_magnitude)
        torch.testing.assert_close(grad, inside, rtol=0, atol=1e-6)
        assert grad.abs().max() > 0.1
    flow = torch.tensor([0.0, 2.0], dtype=torch.float64, device=device).view(1, 2, 1, 1)
    max_magnitude = torch.tensor(2.0, dtype=torch.float64, device=device, requires_grad=True)
    rgb = flow_to_rgb(flow, max_magnitude=max_magnitude).flatten()
    grad = torch.stack([torch.autograd.grad(channel, max_magnitude, retain_graph=True)[0] for channel in rgb])
    inside = (flow_to_rgb(flow, max_magnitude=2.0 + 1e-7).flatten() - rgb.detach()) / 1e-7
    torch.testing.assert_close(grad, inside, rtol=0, atol=1e-6)
    assert grad.abs().max() > 0.1


def test_tiny_flows(device: torch.device, dtype: torch.dtype) -> None:
    """Finite gradients down to the smallest flows; atan2's backward divides by u² + v², which would overflow."""
    smallest = math.log10(torch.finfo(dtype).smallest_normal) - 7
    magnitudes = torch.logspace(smallest, -10, 200, dtype=torch.float64)
    flow = direction_flow(torch.full_like(magnitudes, -0.93), magnitudes).to(device=device, dtype=dtype)
    flow.requires_grad_()
    for max_magnitude in (None, 1.0):
        (grad,) = torch.autograd.grad(flow_to_rgb(flow, max_magnitude=max_magnitude).sum(), flow)
        assert torch.isfinite(grad).all()
    # the gradient is the limit at zero, while the squared length keeps 10 bits as a subnormal
    (grad,) = torch.autograd.grad(flow_to_rgb(flow, max_magnitude=1.0).sum(), flow)
    finfo = torch.finfo(dtype)
    precise = magnitudes.to(device) ** 2 > 1024 * finfo.smallest_normal * finfo.eps
    reference = direction_flow(torch.tensor([-0.93], dtype=torch.float64), 1e-12).to(device).requires_grad_()
    (limit,) = torch.autograd.grad(flow_to_rgb(reference, max_magnitude=1.0).sum(), reference)
    torch.testing.assert_close(grad[..., precise], limit.to(dtype).expand_as(grad[..., precise]), rtol=0, atol=1e-3)


@pytest.mark.parametrize("given", [False, True])
def test_second_derivatives_at_still_pixels(device: torch.device, given: bool) -> None:
    flow = torch.randn(1, 2, 3, 4, device=device, dtype=torch.float64)
    flow[..., 1, 2] = 0
    flow.requires_grad_()
    out = flow_to_rgb(flow, max_magnitude=2.0 if given else None)
    (grad,) = torch.autograd.grad((out * torch.rand_like(out)).sum(), flow, create_graph=True)
    (hessian_vector,) = torch.autograd.grad(grad.square().sum(), flow)
    assert torch.isfinite(hessian_vector).all()


def test_batch_independence(device: torch.device) -> None:
    flow = torch.randn(3, 2, 6, 7, device=device, dtype=torch.float64)
    out = flow_to_rgb(flow)
    for b in range(3):
        torch.testing.assert_close(out[b : b + 1], flow_to_rgb(flow[b : b + 1]), rtol=0, atol=0)


@pytest.mark.parametrize("given", [False, True])
def test_gradcheck(device: torch.device, given: bool) -> None:
    flow = torch.randn(2, 2, 4, 5, device=device, dtype=torch.float64, requires_grad=True)
    if given:
        max_magnitude = torch.tensor(1.5, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(lambda f, m: flow_to_rgb(f, max_magnitude=m), (flow, max_magnitude))
        assert gradgradcheck(lambda f, m: flow_to_rgb(f, max_magnitude=m), (flow, max_magnitude))
    else:
        assert gradcheck(flow_to_rgb, (flow,))
        assert gradgradcheck(flow_to_rgb, (flow,))


@pytest.mark.compile
def test_compile(device: torch.device) -> None:
    torch._dynamo.reset()
    flow = 3 * torch.randn(2, 2, 20, 30, device=device)
    flow[0, :, :4] = 0
    compiled = torch.compile(flow_to_rgb, fullgraph=True)
    torch.testing.assert_close(compiled(flow), flow_to_rgb(flow), rtol=1e-5, atol=1e-5)


def test_argument_errors() -> None:
    flow = torch.randn(1, 2, 4, 5)
    for max_magnitude in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="max_magnitude"):
            flow_to_rgb(flow, max_magnitude=max_magnitude)
    with pytest.raises(TypeError, match="max_magnitude"):
        flow_to_rgb(flow, max_magnitude="1")
    with pytest.raises(ValueError, match="flow must have shape"):
        flow_to_rgb(torch.randn(1, 3, 4, 5))
    with pytest.raises(TypeError, match="floating-point"):
        flow_to_rgb(torch.ones(1, 2, 4, 5, dtype=torch.int64))

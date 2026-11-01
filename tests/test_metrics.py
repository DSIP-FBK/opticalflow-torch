"""Tests of endpoint_error, photometric_error and consistency."""

from __future__ import annotations

import pytest
import torch
from synthetic import AnalyticTexture, make_pair, rotation_flow, translation_flow
from torch import Tensor
from torch.autograd import gradcheck, gradgradcheck

from opticalflow_torch import consistency, endpoint_error, photometric_error, warp

PADDINGS = ["border", "zeros", "reflection"]


def random_flows(
    batch: int = 2,
    height: int = 7,
    width: int = 9,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> tuple[Tensor, Tensor]:
    """Two flows with fractional displacements, away from bilinear kinks."""
    flow = 1.5 * torch.randn(batch, 2, height, width, device=device, dtype=dtype)
    return flow, 1.5 * torch.randn(batch, 2, height, width, device=device, dtype=dtype)


def consistency_reference(forward: Tensor, backward: Tensor, alpha: float, beta: float) -> Tensor:
    back = warp(backward, forward)
    mismatch = ((forward + back) ** 2).sum(1)
    return torch.exp(-mismatch / (alpha * ((forward**2).sum(1) + (back**2).sum(1)) + beta))


# ----------------------------------------------------------------------------------------------
# Endpoint error
# ----------------------------------------------------------------------------------------------


def test_endpoint_error(device: torch.device, dtype: torch.dtype) -> None:
    flow, reference = random_flows(device=device, dtype=dtype)
    out = endpoint_error(flow, reference)
    assert out.shape == (2, 7, 9) and out.dtype == dtype and out.device == flow.device
    torch.testing.assert_close(out, ((flow - reference) ** 2).sum(1).sqrt())
    assert torch.equal(endpoint_error(flow, flow), torch.zeros_like(out))
    flow[0, 0, 1, 2], flow[1, 1, 3, 4] = float("nan"), float("inf")
    out = endpoint_error(flow, reference)
    assert out[0, 1, 2].isnan() and out[1, 3, 4] == float("inf") and out.isfinite().sum() == out.numel() - 2


def test_endpoint_error_gradient_where_equal(device: torch.device, dtype: torch.dtype) -> None:
    flow, reference = random_flows(device=device, dtype=dtype)
    reference[:, :, :3] = flow[:, :, :3]  # equal on the first rows
    flow.requires_grad_()
    reference.requires_grad_()
    grad_flow, grad_reference = torch.autograd.grad(endpoint_error(flow, reference).sum(), (flow, reference))
    assert torch.isfinite(grad_flow).all() and torch.isfinite(grad_reference).all()
    assert torch.equal(grad_flow[:, :, :3], torch.zeros_like(grad_flow[:, :, :3]))
    unit = (flow - reference)[:, :, 3:] / endpoint_error(flow, reference)[:, None, 3:]
    torch.testing.assert_close(grad_flow[:, :, 3:], unit)
    torch.testing.assert_close(grad_reference, -grad_flow)


def test_endpoint_error_gradcheck(device: torch.device) -> None:
    inputs = [x.requires_grad_() for x in random_flows(device=device)]
    assert gradcheck(endpoint_error, inputs)
    assert gradgradcheck(endpoint_error, inputs)


def test_endpoint_error_second_derivatives_where_equal(device: torch.device) -> None:
    """Pixels where the flows are equal add nothing to the derivatives of any order (not NaN)."""
    weight = torch.tensor(0.7, device=device, dtype=torch.float64, requires_grad=True)
    flow, _ = random_flows(device=device)
    reference = 0.7 * flow
    reference[..., 0, :] += 1.0  # equal elsewhere at weight 0.7
    derivatives = []
    for rows in (slice(None), slice(0, 1)):
        (grad,) = torch.autograd.grad(
            endpoint_error(weight * flow, reference)[:, rows].sum(), weight, create_graph=True
        )
        (second,) = torch.autograd.grad(grad, weight)
        derivatives.append(torch.stack((grad, second)))
    assert torch.isfinite(derivatives[0]).all()
    torch.testing.assert_close(derivatives[0], derivatives[1])
    # gradient penalty: the derivative of the squared gradient, with flow = reference = 0
    zeros = torch.zeros(1, 2, 3, 4, device=device, dtype=torch.float64, requires_grad=True)
    (grad,) = torch.autograd.grad(endpoint_error(zeros, torch.zeros_like(zeros)).sum(), zeros, create_graph=True)
    (penalty,) = torch.autograd.grad(grad.square().sum(), zeros)
    assert torch.equal(penalty, torch.zeros_like(penalty))


# ----------------------------------------------------------------------------------------------
# Photometric error
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("padding_mode", PADDINGS)
def test_photometric_error(device: torch.device, dtype: torch.dtype, padding_mode: str) -> None:
    frame0 = torch.rand(2, 3, 7, 9, device=device, dtype=dtype)
    frame1 = torch.rand(2, 3, 7, 9, device=device, dtype=dtype)
    flow, _ = random_flows(device=device, dtype=dtype)
    out = photometric_error(frame0, frame1, flow, padding_mode=padding_mode)
    assert out.shape == (2, 3, 7, 9) and out.dtype == dtype and out.device == frame0.device
    torch.testing.assert_close(out, (warp(frame1, flow, padding_mode=padding_mode) - frame0).abs())


def test_photometric_error_of_an_exact_pair(device: torch.device, dtype: torch.dtype) -> None:
    """An integer translation is warped exactly, inside the frame: no error there."""
    flow = translation_flow(30, 40, 3, -2, batch=2, device=device, dtype=dtype)
    frame0, frame1 = make_pair(AnalyticTexture(4, channels=2), flow)
    out = photometric_error(frame0, frame1, flow)
    tolerance = 1e-3 if dtype == torch.float32 else 1e-10  # 0-255 values
    assert out[..., 2:, :-3].max() < tolerance
    assert out[..., :2, :].max() > 1  # rows read from beyond the frame


@pytest.mark.parametrize("padding_mode", PADDINGS)
def test_photometric_error_gradcheck(device: torch.device, padding_mode: str) -> None:
    frame0 = torch.rand(1, 2, 5, 6, device=device, dtype=torch.float64, requires_grad=True)
    frame1 = torch.rand(1, 2, 5, 6, device=device, dtype=torch.float64, requires_grad=True)
    flow = random_flows(1, 5, 6, device=device)[0].requires_grad_()

    def fn(f0: Tensor, f1: Tensor, fl: Tensor) -> Tensor:
        return photometric_error(f0, f1, fl, padding_mode=padding_mode)

    nondet_tol = 1e-12 if device.type == "cuda" else 0.0  # grid_sample's CUDA backward adds atomically
    assert gradcheck(fn, (frame0, frame1, flow), nondet_tol=nondet_tol)


# ----------------------------------------------------------------------------------------------
# Consistency
# ----------------------------------------------------------------------------------------------


def test_consistency_of_inverse_translations(device: torch.device, dtype: torch.dtype) -> None:
    forward = translation_flow(20, 30, 3.3, -2.1, batch=2, device=device, dtype=dtype)
    out = consistency(forward, -forward)
    assert out.shape == (2, 20, 30) and out.dtype == dtype and out.device == forward.device
    assert (out > 1 - 1e-6).all() and (out <= 1).all()


def test_consistency_of_inverse_rotations(device: torch.device, dtype: torch.dtype) -> None:
    """Bilinear interpolation of an affine backward flow is exact, except where it reads the border."""
    forward = rotation_flow(40, 50, 6.0, device=device, dtype=dtype)
    backward = rotation_flow(40, 50, -6.0, device=device, dtype=dtype)
    out = consistency(forward, backward)
    assert (out[:, 5:-5, 5:-5] > 1 - 1e-6).all()
    assert out.min() < 0.99  # corners leave the frame


def test_consistency_values(device: torch.device, dtype: torch.dtype) -> None:
    forward, backward = random_flows(device=device, dtype=dtype)
    torch.testing.assert_close(consistency(forward, backward), consistency_reference(forward, backward, 0.01, 0.5))
    out = consistency(forward, backward, alpha=0.2, beta=torch.tensor(1.5, device=device, dtype=dtype))
    torch.testing.assert_close(out, consistency_reference(forward, backward, 0.2, 1.5))
    assert ((out > 0) & (out <= 1)).all()
    # the same displacement both ways instead of opposite ones: |f + b|^2 = 4 |f|^2
    same = translation_flow(5, 6, 3.0, 0.0, device=device, dtype=dtype)
    expected = torch.exp(torch.tensor(-36 / (0.01 * 18 + 0.5), dtype=torch.float64)).to(dtype)
    torch.testing.assert_close(consistency(same, same), expected.expand(1, 5, 6).to(device))


def test_consistency_decreases_with_the_mismatch(device: torch.device) -> None:
    forward = translation_flow(8, 8, 2.0, 1.0, device=device)
    values = [consistency(forward, -forward + error).mean() for error in (0.0, 0.3, 0.6, 1.2)]
    assert all(a > b for a, b in zip(values, values[1:], strict=False))


def test_consistency_underflows_to_zero(device: torch.device, dtype: torch.dtype) -> None:
    """Large mismatches give exactly 0, within the documented [0, 1], with zero gradient."""
    same = translation_flow(4, 5, 10.0, 0.0, device=device, dtype=dtype).requires_grad_()
    out = consistency(same, same, alpha=0.01 if dtype == torch.float32 else 0.0)
    assert torch.equal(out, torch.zeros_like(out))
    (grad,) = torch.autograd.grad(out.sum(), same)
    assert torch.equal(grad, torch.zeros_like(grad))


def test_consistency_zero_flows(device: torch.device) -> None:
    forward = torch.zeros(1, 2, 6, 7, device=device, requires_grad=True)
    backward = torch.zeros(1, 2, 6, 7, device=device, requires_grad=True)
    out = consistency(forward, backward)
    assert torch.equal(out, torch.ones_like(out))
    grads = torch.autograd.grad(out.sum(), (forward, backward))
    assert all(torch.isfinite(g).all() for g in grads)


def test_consistency_gradcheck(device: torch.device) -> None:
    forward, backward = random_flows(1, 5, 6, device=device)
    alpha = torch.tensor(0.05, device=device, dtype=torch.float64)
    beta = torch.tensor(0.8, device=device, dtype=torch.float64)
    inputs = [x.requires_grad_() for x in (forward, backward, alpha, beta)]

    def fn(f: Tensor, b: Tensor, a: Tensor, c: Tensor) -> Tensor:
        return consistency(f, b, alpha=a, beta=c)

    nondet_tol = 1e-12 if device.type == "cuda" else 0.0  # grid_sample's CUDA backward adds atomically
    assert gradcheck(fn, inputs, nondet_tol=nondet_tol)


# ----------------------------------------------------------------------------------------------
# Shared
# ----------------------------------------------------------------------------------------------


def test_batch_independence(device: torch.device) -> None:
    forward, backward = random_flows(3, device=device)
    frame0 = torch.rand(3, 2, 7, 9, device=device, dtype=torch.float64)
    frame1 = torch.rand(3, 2, 7, 9, device=device, dtype=torch.float64)
    outs = (
        endpoint_error(forward, backward),
        photometric_error(frame0, frame1, forward),
        consistency(forward, backward),
    )
    for b in range(3):
        one = slice(b, b + 1)
        singles = (
            endpoint_error(forward[one], backward[one]),
            photometric_error(frame0[one], frame1[one], forward[one]),
            consistency(forward[one], backward[one]),
        )
        for out, single in zip(outs, singles, strict=True):
            torch.testing.assert_close(out[one], single, rtol=0, atol=1e-14)


@pytest.mark.compile
def test_compile(device: torch.device) -> None:
    torch._dynamo.reset()
    forward, backward = random_flows(device=device, dtype=torch.float32)
    frame0 = torch.rand(2, 1, 7, 9, device=device)
    frame1 = torch.rand(2, 1, 7, 9, device=device)

    def fn(f0: Tensor, f1: Tensor, f: Tensor, b: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return endpoint_error(f, b), photometric_error(f0, f1, f), consistency(f, b)

    compiled = torch.compile(fn, fullgraph=True)
    for out, ref in zip(
        compiled(frame0, frame1, forward, backward), fn(frame0, frame1, forward, backward), strict=True
    ):
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_argument_errors() -> None:
    forward, backward = random_flows()
    frame = torch.rand(2, 1, 7, 9, dtype=torch.float64)
    with pytest.raises(ValueError, match="reference must have shape"):
        endpoint_error(forward, backward[:, :, 1:])
    with pytest.raises(TypeError, match="reference must have the dtype"):
        endpoint_error(forward, backward.float())
    with pytest.raises(ValueError, match="flow must have shape"):
        endpoint_error(forward[:, :1], backward[:, :1])
    with pytest.raises(ValueError, match="same shape"):
        photometric_error(frame, frame[:1], forward)
    with pytest.raises(ValueError, match="flow must have shape"):
        photometric_error(frame, frame, forward[:1])
    with pytest.raises(ValueError, match="padding_mode"):
        photometric_error(frame, frame, forward, padding_mode="wrap")
    with pytest.raises(ValueError, match="backward_flow must have shape"):
        consistency(forward, backward[:1])
    with pytest.raises(TypeError, match="forward_flow must be a torch.Tensor"):
        consistency(forward.numpy(), backward)
    with pytest.raises(ValueError, match="alpha"):
        consistency(forward, backward, alpha=-0.1)
    for beta in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="beta"):
            consistency(forward, backward, beta=beta)
    with pytest.raises(TypeError, match="beta"):
        consistency(forward, backward, beta="0.5")

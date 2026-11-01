"""Tests of the high-level API: optical_flow, interpolate and OpticalFlow."""

from __future__ import annotations

import pytest
import torch
from synthetic import AnalyticTexture, make_pair, rotation_flow
from torch import Tensor

from opticalflow_torch import (
    Farneback,
    LucasKanade,
    OpticalFlow,
    farneback,
    interpolate,
    interpolate_frames,
    lucas_kanade,
    optical_flow,
)

METHODS = {"farneback": (farneback, Farneback), "lucas_kanade": (lucas_kanade, LucasKanade)}
OPTIONS = {"levels": 1, "iterations": 2, "window": "gaussian"}  # small and valid for both methods


def pair(device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> tuple[Tensor, Tensor]:
    """Frames in 0-255 related by a rotation plus a shift, batch of 2, two channels."""
    flow = rotation_flow(40, 44, 3.0, batch=2, device=device, dtype=dtype) + 1.0
    return make_pair(AnalyticTexture(1, channels=2), flow)


@pytest.mark.parametrize("method", METHODS)
def test_matches_the_method(method: str) -> None:
    frame0, frame1 = pair()
    function, module = METHODS[method]
    initial = torch.full((2, 2, 40, 44), 0.5)
    options = {**OPTIONS, "eps": 0.01}
    expected = function(frame0, frame1, initial_flow=initial, return_confidence=True, **options)
    out = optical_flow(frame0, frame1, method=method, initial_flow=initial, return_confidence=True, **options)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    model, reference = OpticalFlow(method, **options, learnable=True), module(**options, learnable=True)
    assert model.method == method and repr(model.estimator) == repr(reference)
    assert f"method={method!r}" in repr(model)
    expected = reference(frame0, frame1, initial, True)
    torch.testing.assert_close(model(frame0, frame1, initial, True), expected, rtol=0, atol=0)
    assert isinstance(OpticalFlow(method=method).estimator, module)


def test_default_method_is_farneback() -> None:
    frame0, frame1 = pair()
    assert torch.equal(optical_flow(frame0, frame1, levels=1), farneback(frame0, frame1, levels=1))
    assert OpticalFlow().method == "farneback" and isinstance(OpticalFlow().estimator, Farneback)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("bidirectional", [False, True])
def test_interpolate_is_interpolate_frames(method: str, bidirectional: bool) -> None:
    frame0, frame1 = pair()
    model = OpticalFlow(method, **OPTIONS)
    flow = model(frame0, frame1)
    backward_flow = model(frame1, frame0) if bidirectional else None
    options = {"bidirectional": bidirectional, "match_distribution": True, "quantiles": 32}
    kwargs = {"backward_flow": backward_flow, "match_distribution": True, "quantiles": 32}
    expected = interpolate_frames(frame0, frame1, flow, [1 / 3, 2 / 3], **kwargs)
    out = interpolate(frame0, frame1, steps=2, method=method, **OPTIONS, **options)
    assert out.shape == (2, 2, 2, 40, 44)
    assert torch.equal(out, expected)
    assert torch.equal(model.interpolate(frame0, frame1, steps=2, **options), expected)
    times = torch.tensor([0.0, 0.25, 1.0])
    expected = interpolate_frames(frame0, frame1, flow, times, backward_flow=backward_flow)
    assert torch.equal(model.interpolate(frame0, frame1, t=times, bidirectional=bidirectional), expected)
    for steps, t in [(1, 0.5), (3, [0.25, 0.5, 0.75]), (4, [0.2, 0.4, 0.6, 0.8])]:
        assert torch.equal(model.interpolate(frame0, frame1, steps=steps), model.interpolate(frame0, frame1, t=t))


def test_argument_errors() -> None:
    frame0, frame1 = pair()
    module = OpticalFlow(**OPTIONS)
    runs = (
        lambda **kwargs: interpolate(frame0, frame1, **OPTIONS, **kwargs),
        lambda **kwargs: module.interpolate(frame0, frame1, **kwargs),
    )
    for run in runs:
        for kwargs in ({}, {"steps": 2, "t": [0.5]}):
            with pytest.raises(ValueError, match="exactly one of steps and t"):
                run(**kwargs)
        for steps in (0, -1):
            with pytest.raises(ValueError, match="steps must be at least 1"):
                run(steps=steps)
        for steps in (True, 2.0, "2"):
            with pytest.raises(TypeError, match="steps must be an integer"):
                run(steps=steps)
    for name, value in (("initial_flow", torch.zeros(2, 2, 40, 44)), ("return_confidence", True)):
        with pytest.raises(TypeError, match=f"interpolate does not take {name}"):
            interpolate(frame0, frame1, steps=2, **{name: value})
    with pytest.raises(TypeError):  # the flow function rejects unknown options
        interpolate(frame0, frame1, steps=2, poly_n=5, method="lucas_kanade")
    with pytest.raises(TypeError):  # the module form takes no flow options
        module.interpolate(frame0, frame1, steps=2, levels=1)
    match = r"method must be one of \('farneback', 'lucas_kanade'\)"
    for method in ("Farneback", "horn_schunck", "", None, ["farneback"]):
        with pytest.raises(ValueError, match=match):
            optical_flow(frame0, frame1, method=method)
        with pytest.raises(ValueError, match=match):
            interpolate(frame0, frame1, steps=1, method=method)
        with pytest.raises(ValueError, match=match):
            OpticalFlow(method)


@pytest.mark.parametrize("method", METHODS)
def test_learnable_parameters_train(device: torch.device, method: str) -> None:
    frame0, frame1 = (frame.requires_grad_() for frame in pair(device))
    model = OpticalFlow(method, **OPTIONS, learnable=True).to(device)
    names = {"estimator.log_window_sigma", "estimator.log_eps"}
    names |= {"estimator.log_poly_sigma"} if method == "farneback" else set()
    assert {name for name, _ in model.named_parameters()} == names == set(model.state_dict())
    out = model.interpolate(frame0, frame1, steps=2, bidirectional=True, match_distribution=True)
    grads = torch.autograd.grad(out.square().mean(), [*model.parameters(), frame0, frame1])
    assert all(torch.isfinite(grad).all() and grad.abs().amax() > 0 for grad in grads)
    grads = torch.autograd.grad(model(frame0, frame1).square().mean(), list(model.parameters()))
    assert all(torch.isfinite(grad) and grad != 0 for grad in grads)
    # the state carries over to the method's own module
    reference = METHODS[method][1](**OPTIONS, learnable=True).to(device)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter += 0.3
    reference.load_state_dict({name.removeprefix("estimator."): value for name, value in model.state_dict().items()})
    assert torch.equal(model(frame0, frame1), reference(frame0, frame1))


def test_module_dtype_and_device(device: torch.device) -> None:
    frame0, frame1 = pair(device, torch.float64)
    model = OpticalFlow(**OPTIONS, learnable=True).double().to(device)
    assert all(p.dtype == torch.float64 and p.device.type == device.type for p in model.parameters())
    out = model.interpolate(frame0, frame1, steps=2)
    assert out.dtype == torch.float64 and out.device == frame0.device
    reference = Farneback(**OPTIONS, learnable=True).to(device, torch.float64)
    assert torch.equal(model(frame0, frame1), reference(frame0, frame1))


@pytest.mark.compile
def test_compile(device: torch.device) -> None:
    torch._dynamo.reset()
    frame0, frame1 = pair(device)
    model = OpticalFlow("lucas_kanade", levels=0, iterations=2, learnable=True).to(device)
    compiled = torch.compile(model.interpolate, fullgraph=True)
    out = compiled(frame0, frame1, steps=2)
    expected = model.interpolate(frame0, frame1, steps=2)
    # values in 0-255 in float32; compiled kernels round differently
    torch.testing.assert_close(out, expected, rtol=0, atol=1e-3)
    grads = torch.autograd.grad(out.mean(), list(model.parameters()))
    expected_grads = torch.autograd.grad(expected.mean(), list(model.parameters()))
    torch.testing.assert_close(grads, expected_grads, rtol=1e-3, atol=0)

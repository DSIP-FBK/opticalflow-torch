"""Farneback dense optical flow, ported from OpenCV's ``calcOpticalFlowFarneback``."""

from __future__ import annotations

import math
import operator
from collections.abc import Iterable
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ._ops import (
    as_scalar,
    check_flow,
    check_frames,
    gaussian_kernel1d,
    no_autocast,
    pad_border,
    pyramid_level,
    pyramid_levels,
    resize,
    separable_filter,
)

_WINDOWS = ("box", "gaussian")
_LEARNABLE = ("poly_sigma", "window_sigma", "eps")
_FLT_EPSILON = 1.1920928955078125e-07
_BORDER = 5
_BORDER_WEIGHTS = (0.14, 0.14, 0.4472, 0.4472, 0.4472)  # FarnebackUpdateMatrices, outermost pixel first


# ----------------------------------------------------------------------------------------------
# Argument checks
# ----------------------------------------------------------------------------------------------


def _check_int(value: object, name: str, minimum: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got a bool")
    try:
        value = operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}") from None
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return value


def _check_number(value: object, name: str, *, tensor: bool, positive: bool = False) -> None:
    """Check a finite real number, or with ``tensor`` also a 1-element floating-point tensor."""
    if tensor and isinstance(value, Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must be a scalar, got a tensor of shape {tuple(value.shape)}")
        if not value.is_floating_point():
            raise TypeError(f"{name} must have a floating-point dtype, got {value.dtype}")
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        kind = "a float or a tensor" if tensor else "a float"
        raise TypeError(f"{name} must be {kind}, got {type(value).__name__}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _check_options(
    pyr_scale: float,
    levels: int,
    winsize: int,
    iterations: int,
    poly_n: int,
    poly_sigma: float | Tensor,
    window: str,
    window_sigma: float | Tensor | None,
    eps: float | Tensor,
    data_range: float,
) -> tuple[float, int, int, int, int]:
    """Validate the options of :func:`farneback`; returns ``pyr_scale, levels, winsize, iterations, poly_n``."""
    _check_number(pyr_scale, "pyr_scale", tensor=False)
    if not 0 < pyr_scale < 1:
        raise ValueError(f"pyr_scale must be in (0, 1), got {pyr_scale}")
    levels = _check_int(levels, "levels", 0)
    winsize = _check_int(winsize, "winsize", 1)
    iterations = _check_int(iterations, "iterations", 1)
    poly_n = _check_int(poly_n, "poly_n", 1)
    _check_number(poly_sigma, "poly_sigma", tensor=True)
    if window not in _WINDOWS:
        raise ValueError(f"window must be one of {_WINDOWS}, got {window!r}")
    if window_sigma is not None:
        if window != "gaussian":
            raise ValueError("window_sigma is only valid with window='gaussian'")
        _check_number(window_sigma, "window_sigma", tensor=True, positive=True)
    _check_number(eps, "eps", tensor=True, positive=True)
    _check_number(data_range, "data_range", tensor=False, positive=True)
    return float(pyr_scale), levels, winsize, iterations, poly_n


# ----------------------------------------------------------------------------------------------
# Steps of the algorithm
# ----------------------------------------------------------------------------------------------


def _poly_kernels(n: int, sigma: float | Tensor, like: Tensor) -> tuple[Tensor, Tensor]:
    """FarnebackPrepareGaussian, folded into the 1-D kernels of FarnebackPolyExp.

    Returns 3 vertical kernels, shape ``(3, 2n + 1)``, and 5 horizontal ones, shape
    ``(5, 2n + 1)``, for the coefficients ``(ry, rx, ryy, rxx, rxy)`` in OpenCV's order; the
    horizontal ones read the vertical outputs ``(1, 0, 2, 0, 1)``. Built in double precision
    (except on MPS) from a differentiable ``sigma``.
    """
    table = torch.float32 if like.device.type == "mps" else torch.float64
    if not isinstance(sigma, Tensor) and sigma < _FLT_EPSILON:
        sigma = n * 0.3  # OpenCV's default
    # a tensor is clamped instead, which keeps the flow continuous in it
    sigma = as_scalar(sigma, torch.empty(0, dtype=table, device=like.device), "poly_sigma").clamp(min=_FLT_EPSILON)
    x = torch.arange(-n, n + 1, dtype=table, device=like.device)
    xx = x * x
    g = torch.exp(-xx / (2 * sigma * sigma))
    g = g / g.sum()
    # OpenCV inverts the Gram matrix of the basis (1, x, y, x^2, y^2, xy) under the weights
    # g(x) g(y). With m2 = sum x^2 g, m4 = sum x^4 g and v = m4 - m2^2 (g sums to 1), the entries
    # it uses are ig11 = 1 / m2, ig03 = -m2 / v, ig33 = 1 / v and ig55 = 1 / m2^2. These overflow
    # for small sigma, where g(x != 0) underflows, so they enter as ratios of p = g / g(1) (x != 0),
    # with ig11 moved to the vertical kernel of ry and ig55 split between the two kernels of rxy.
    p = torch.exp(-(xx - 1).clamp(min=0) / (2 * sigma * sigma))
    s2, s4 = (xx * p).sum(), (xx * xx * p).sum()  # m2 / g(1), m4 / g(1)
    derivative = x * p / s2  # ig11 * x g, and sqrt(ig55) * x g
    quadratic = (xx * p - s2 * g) / (s4 - g[n + 1] * s2 * s2)  # ig03 * g + ig33 * x^2 g
    vertical = torch.stack((g, derivative, quadratic))
    horizontal = torch.stack((g, derivative, g, quadratic, derivative))
    return vertical.to(like.dtype), horizontal.to(like.dtype)


def _poly_exp(image: Tensor, kernels: tuple[Tensor, Tensor]) -> Tensor:
    """FarnebackPolyExp of ``(N, 1, H, W)`` images: coefficients ``(ry, rx, ryy, rxx, rxy)``, shape ``(N, 5, H, W)``."""
    vertical, horizontal = kernels
    n = vertical.shape[-1] // 2
    # each coefficient's own vertical pass (indexing the 3 columns with a list would sync with the device)
    vertical = torch.stack([vertical[i] for i in (1, 0, 2, 0, 1)])
    padded = pad_border(image.expand(-1, 5, -1, -1), (n, n, n, n), "replicate")
    columns = F.conv2d(padded, vertical.reshape(5, 1, -1, 1), groups=5)
    return F.conv2d(columns, horizontal.reshape(5, 1, 1, -1), groups=5)


def _border_scale(height: int, width: int, like: Tensor) -> Tensor:
    """FarnebackUpdateMatrices' weights of the pixels near the borders, shape ``(H, W)``."""
    weights = torch.tensor(_BORDER_WEIGHTS, dtype=torch.float32).to(like.device, non_blocking=True)

    def axis(n: int) -> tuple[Tensor, Tensor, Tensor]:
        i = torch.arange(n, device=like.device)
        # OpenCV's (unsigned)(i - 5) >= (unsigned)(n - 10), which differs from i < 5 || i >= n - 5 for n < 10
        near = ((i - _BORDER) & 0xFFFFFFFF) >= ((n - 2 * _BORDER) & 0xFFFFFFFF)
        low = torch.where(i < _BORDER, weights[i.clamp(max=_BORDER - 1)], 1.0)
        high = torch.where(i >= n - _BORDER, weights[(n - 1 - i).clamp(0, _BORDER - 1)], 1.0)
        return near, low, high

    near_y, top, bottom = axis(height)
    near_x, left, right = axis(width)
    scale = (left * right) * top.unsqueeze(-1) * bottom.unsqueeze(-1)  # OpenCV's float product order
    return torch.where(near_y.unsqueeze(-1) | near_x, scale, 1.0).to(like.dtype)


def _inside(coord: Tensor, size: int) -> Tensor:
    """OpenCV's test ``0 <= floor(coord) < size - 1``, linearly interpolated between integer ``coord``."""
    if size < 2:
        return torch.zeros_like(coord)
    return torch.minimum(coord + 1, (size - 1) - coord).clamp(0, 1)


def _index(coord: Tensor, size: int) -> Tensor:
    return torch.nan_to_num(coord, nan=0.0).clamp(0, size - 1).long()


def _update_matrices(r0: Tensor, r1: tuple[Tensor, Tensor], flow: Tensor, scale: Tensor) -> Tensor:
    """FarnebackUpdateMatrices: each pixel's ``(g11, g12, g22, h1, h2)``, summed over channels.

    ``r0`` holds ``(B, C, 5, H, W)`` coefficients, ``rxy`` halved; ``r1`` is from :func:`_prepare`,
    ``flow`` is ``(B, 2, H, W)`` and ``scale`` is half the border weights. Returns shape
    ``(B, 5, H, W)``. Where OpenCV switches between the bilinear sample of ``r1`` at the displaced
    position and a fallback on ``r0``, this blends the two with :func:`_inside`, so it equals
    OpenCV wherever that weight is 0 or 1.
    """
    batch, channels, _, height, width = r0.shape
    padded, fallback = r1
    dx, dy = flow[:, 0], flow[:, 1]
    fx = torch.arange(width, dtype=flow.dtype, device=flow.device) + dx
    fy = torch.arange(height, dtype=flow.dtype, device=flow.device).unsqueeze(-1) + dy
    x0, y0 = fx.floor(), fy.floor()
    ax, ay = (fx - x0).reshape(batch, 1, -1), (fy - y0).reshape(batch, 1, -1)
    # top-left tap in r1 padded by a replicated pixel: clamped to [0, size], all 4 taps read OpenCV's clamped ones
    stride = width + 2
    top = torch.add(_index(x0 + 1, width + 1), _index(y0 + 1, height + 1), alpha=stride).reshape(batch, 1, -1)

    def tap(offset: int) -> Tensor:
        return padded.gather(2, (top + offset if offset else top).expand(-1, padded.shape[1], -1))

    bx, by = 1 - ax, 1 - ay
    sample = bx * by * tap(0) + ax * by * tap(1) + bx * ay * tap(stride) + ax * ay * tap(stride + 1)
    sample = sample.reshape(batch, channels, 5, height, width)
    valid = (_inside(fx, width) * _inside(fy, height)).reshape(batch, 1, 1, height, width)
    # doubled coefficients; outside, OpenCV takes 0 for r1's linear ones and r0's for the quadratic ones
    zeros_linear, sign = fallback
    r2, r3, r4, r5, r6 = torch.addcmul(r0, torch.lerp(zeros_linear, sample, valid), sign).unbind(2)
    dx, dy = dx.unsqueeze(1), dy.unsqueeze(1)
    r2 = (r2 + (r4 * dy + r6 * dx)) * scale
    r3 = (r3 + (r6 * dy + r5 * dx)) * scale
    r4, r5, r6 = r4 * scale, r5 * scale, r6 * scale
    r66 = r6 * r6
    matrices = (r4 * r4 + r66, (r4 + r5) * r6, r5 * r5 + r66, r4 * r2 + r6 * r3, r6 * r2 + r5 * r3)
    return torch.cat(matrices, 1) if channels == 1 else torch.stack(matrices, dim=1).sum(2)


def _prepare(r0: Tensor, r1: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    """Per-level inputs of :func:`_update_matrices` from ``(B, C, 5, H, W)`` coefficients.

    Returns ``r1`` padded by a replicated pixel, flattened to ``(B, 5C, (H + 2) (W + 2))``, and
    ``r0`` with zeroed linear coefficients and the signs that combine it with the sample.
    """
    batch, channels, _, height, width = r1.shape
    padded = pad_border(r1.reshape(batch, channels * 5, height, width), (1, 1, 1, 1), "replicate")
    sign = torch.tensor([-1.0, -1.0, 1.0, 1.0, 1.0], dtype=r0.dtype).to(r0.device, non_blocking=True).reshape(5, 1, 1)
    return padded.reshape(batch, channels * 5, -1), (r0 * (sign > 0), sign)


def _window_kernel(window: str, winsize: int, window_sigma: float | Tensor | None, like: Tensor) -> Tensor:
    """1-D averaging kernel of FarnebackUpdateFlow_Blur or FarnebackUpdateFlow_GaussianBlur."""
    radius = winsize // 2
    if window == "box":
        # OpenCV divides sums over 2 * radius + 1 pixels by winsize ** 2, also for even winsize
        return torch.full((2 * radius + 1,), 1 / winsize, dtype=like.dtype, device=like.device)
    if radius == 0:
        return torch.ones(1, dtype=like.dtype, device=like.device)
    sigma = 0.3 * radius if window_sigma is None else window_sigma
    return gaussian_kernel1d(as_scalar(sigma, like, "window_sigma"), radius)


def _solve(averaged: Tensor, eps: Tensor) -> Tensor:
    """Flow from the window-averaged ``(g11, g12, g22, h1, h2)``, as FarnebackUpdateFlow_*."""
    g11, g12, g22, h1, h2 = averaged.unbind(1)
    idet = 1 / (g11 * g22 - g12 * g12 + eps)
    return torch.stack(((g11 * h2 - g12 * h1) * idet, (g22 * h1 - g12 * h2) * idet), dim=1)


def _min_eigenvalue(averaged: Tensor) -> Tensor:
    """Smallest eigenvalue of the symmetric systems ``(g11, g12, g22)`` in ``averaged``, shape ``(B, H, W)``."""
    g11, g12, g22 = averaged[:, 0], averaged[:, 1], averaged[:, 2]
    # The radius sqrt(r2) has unbounded derivatives at r2 = 0, which flat regions reach exactly;
    # sqrt(r2 + c) - sqrt(c) has bounded ones and is within sqrt(c) (3e-10 in float32) of it. Written
    # without the subtraction, it is exactly 0 at r2 = 0 however sqrt rounds (AVX-512 differs).
    c = torch.finfo(averaged.dtype).tiny ** 0.5
    r2 = ((g11 - g22) * 0.5).square() + g12.square()
    radius = r2 / ((r2 + c).sqrt() + math.sqrt(c))
    return (g11 + g22) * 0.5 - radius


# ----------------------------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------------------------


@no_autocast
def farneback(
    frame0: Tensor,
    frame1: Tensor,
    *,
    pyr_scale: float = 0.5,
    levels: int = 5,
    winsize: int = 13,
    iterations: int = 10,
    poly_n: int = 5,
    poly_sigma: float | Tensor = 1.1,
    window: Literal["box", "gaussian"] = "box",
    window_sigma: float | Tensor | None = None,
    eps: float | Tensor = 1e-3,
    data_range: float = 255.0,
    initial_flow: Tensor | None = None,
    return_confidence: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Dense optical flow from ``frame0`` to ``frame1`` with Farneback's algorithm.

    A port of OpenCV's ``cv2.calcOpticalFlowFarneback``, differentiable with respect to
    both frames, ``initial_flow``, ``poly_sigma``, ``window_sigma`` and ``eps``. Defaults
    are OpenCV's ``FarnebackOpticalFlow.create`` defaults, which assume frames spanning 0-255;
    give ``data_range`` for others, e.g. ``data_range=1.0`` for frames in [0, 1]. Frames with
    several channels sum their per-channel equations. float16 and bfloat16 frames are processed
    in float32, and float32 frames stay in float32 inside :func:`torch.autocast`.

    Intended differences: OpenCV's test of whether a displaced position falls inside the image
    becomes a weight that varies linearly between integer positions, which keeps the flow
    continuous, so results differ from OpenCV's near the borders; and small ``poly_sigma``
    values for which OpenCV returns a zero flow work here (see ``poly_sigma``).

    Parameters
    ----------
    frame0, frame1 : Tensor
        Frames, shape ``(B, C, H, W)``, floating point.
    pyr_scale : float
        Size ratio between consecutive pyramid levels, below 1.
    levels : int
        Number of coarser levels added on top of the full-resolution frames; 0 means no
        pyramid. This follows OpenCV's code (its documentation counts the full-resolution
        level too). Levels smaller than 32 pixels are skipped, as in OpenCV.
    winsize : int
        Size of the window over which each pixel's equations are averaged. OpenCV's box window
        also adds the first row and column when ``winsize=1``; that slip is not reproduced.
    iterations : int
        Iterations at each pyramid level, at least 1.
    poly_n : int
        Half-size of the neighbourhood of the polynomial expansion, typically 5 or 7.
    poly_sigma : float or Tensor
        Gaussian sigma of the polynomial expansion weights. A number below float32's epsilon
        means ``0.3 * poly_n``, as in OpenCV; a tensor is clamped to at least that epsilon.
        Between it and about 0.164 OpenCV fails to invert its moment matrix and returns a zero
        flow; here the expansion tends to central differences as ``poly_sigma`` decreases.
    window : {"box", "gaussian"}
        Averaging window; ``"gaussian"`` is OpenCV's ``OPTFLOW_FARNEBACK_GAUSSIAN``.
    window_sigma : float or Tensor, optional
        Sigma of the Gaussian window, by default ``0.3 * (winsize // 2)`` as in OpenCV.
        Only valid with ``window="gaussian"``.
    eps : float or Tensor
        Added to the determinant of each 2x2 system of frames spanning 0-255 (OpenCV uses
        1e-3); for frames spanning ``data_range``, ``eps * (data_range / 255) ** 4``, as the
        determinant scales with the fourth power of the values. Must be positive: with 0,
        flat regions give NaN.
    data_range : float
        Span of the frame values, positive: 255 for 0-255 frames, 1.0 for frames in [0, 1].
        Frames scaled together with it give the same flow. Too large a value shortens the
        flow, to about zero for frames in [0, 1] left at 255.
    initial_flow : Tensor, optional
        Starting estimate, shape ``(B, 2, H, W)`` (OpenCV's ``OPTFLOW_USE_INITIAL_FLOW``).
    return_confidence : bool
        Also return the smallest eigenvalue of each pixel's averaged system at full
        resolution, shape ``(B, H, W)``. It scales with the square of the frame values, not
        with ``data_range``.

    Returns
    -------
    Tensor or tuple of Tensor
        Flow of shape ``(B, 2, H, W)`` in pixels, and the confidence if requested.
    """
    check_frames(frame0, frame1)
    pyr_scale, levels, winsize, iterations, poly_n = _check_options(
        pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, window, window_sigma, eps, data_range
    )
    if initial_flow is not None:
        check_flow(initial_flow, frame0, "initial_flow")
    dtype = frame0.dtype
    compute = torch.promote_types(dtype, torch.float32)
    batch, channels, height, width = frame0.shape
    # In units of 0-255 frames, where eps applies: the same as eps * (data_range / 255) ** 4 on the
    # frames as given, whose float32 gradients broke down from data_range 2.5e7 (1 / det ** 2 underflows).
    scale = 255 / data_range
    frames = torch.cat((frame0, frame1)).to(compute) * scale
    vertical, horizontal = _poly_kernels(poly_n, poly_sigma, frames)
    # rxy halved, exactly, as FarnebackUpdateMatrices uses it; scale = border weights / 2 halves the rest
    poly_kernels = (vertical, torch.cat((horizontal[:4], horizontal[4:] * 0.5)))
    window_kernel = _window_kernel(window, winsize, window_sigma, frames)
    eps = as_scalar(eps, frames, "eps")

    pyramid = pyramid_levels(height, width, pyr_scale, levels)
    flow = None if initial_flow is None else initial_flow.to(compute)
    for k in range(len(pyramid) - 1, -1, -1):
        level = pyramid[k]
        size = (level.height, level.width)
        if k < len(pyramid) - 1:
            flow = resize(flow, size, "linear") * (1 / pyr_scale)
        elif flow is None:
            flow = frames.new_zeros(batch, 2, *size)
        else:
            flow = resize(flow, size, "area") * level.scale
        images = pyramid_level(frames, level).reshape(2 * batch * channels, 1, *size)
        r0, r1 = _poly_exp(images, poly_kernels).reshape(2, batch, channels, 5, *size).unbind(0)
        border_scale = _border_scale(*size, frames) * 0.5
        r1 = _prepare(r0, r1)
        matrices = _update_matrices(r0, r1, flow, border_scale)
        for i in range(iterations):
            averaged = separable_filter(matrices, window_kernel, window_kernel, border="replicate")
            flow = _solve(averaged, eps)
            if i < iterations - 1:
                matrices = _update_matrices(r0, r1, flow, border_scale)
    if return_confidence:
        return flow.to(dtype), (_min_eigenvalue(averaged) / scale**2).to(dtype)
    return flow.to(dtype)


def _learnable_names(learnable: bool | Iterable[str], window: str, winsize: int) -> set[str]:
    has_window_sigma = window == "gaussian" and winsize >= 2
    if isinstance(learnable, bool):
        return {n for n in _LEARNABLE if n != "window_sigma" or has_window_sigma} if learnable else set()
    if isinstance(learnable, str):
        learnable = [learnable]
    try:
        names = list(learnable)
    except TypeError:
        raise TypeError(f"learnable must be a bool or an iterable of str, got {type(learnable).__name__}") from None
    for name in names:
        if name not in _LEARNABLE:
            raise ValueError(f"learnable parameters must be among {_LEARNABLE}, got {name!r}")
    if "window_sigma" in names and not has_window_sigma:
        raise ValueError("window_sigma can only be learned with window='gaussian' and winsize >= 2")
    return set(names)


class Farneback(nn.Module):
    """Module form of :func:`farneback`, with optionally learnable parameters.

    Parameters are those of :func:`farneback` except the frames, ``initial_flow`` and
    ``return_confidence``, which go to :meth:`forward`.

    Parameters
    ----------
    learnable : bool or iterable of str
        Parameters to register as :class:`torch.nn.Parameter`, among ``"poly_sigma"``,
        ``"window_sigma"`` (Gaussian window only) and ``"eps"``; ``True`` means all that
        apply. They are stored as logarithms so they stay positive. ``eps`` stays in units of
        0-255 frames, so a learned value carries over to another ``data_range``, which is not
        learnable.
    """

    def __init__(
        self,
        *,
        pyr_scale: float = 0.5,
        levels: int = 5,
        winsize: int = 13,
        iterations: int = 10,
        poly_n: int = 5,
        poly_sigma: float = 1.1,
        window: Literal["box", "gaussian"] = "box",
        window_sigma: float | None = None,
        eps: float = 1e-3,
        data_range: float = 255.0,
        learnable: bool | Iterable[str] = False,
    ) -> None:
        super().__init__()
        for name, value in (("poly_sigma", poly_sigma), ("window_sigma", window_sigma), ("eps", eps)):
            if value is not None:
                _check_number(value, name, tensor=False)
        self.pyr_scale, self.levels, self.winsize, self.iterations, self.poly_n = _check_options(
            pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, window, window_sigma, eps, data_range
        )
        self.window = window
        self.data_range = float(data_range)
        names = _learnable_names(learnable, window, self.winsize)
        if window_sigma is None and "window_sigma" in names:
            window_sigma = 0.3 * (self.winsize // 2)
        self._fixed: dict[str, float | None] = {}
        for name, value in (("poly_sigma", poly_sigma), ("window_sigma", window_sigma), ("eps", eps)):
            if name not in names:
                self._fixed[name] = None if value is None else float(value)
                continue
            if not value > 0:
                raise ValueError(f"{name} must be positive to be learnable, got {value}")
            self.register_parameter(f"log_{name}", nn.Parameter(torch.tensor(math.log(value))))

    def _value(self, name: str) -> float | Tensor | None:
        if name in self._fixed:
            return self._fixed[name]
        return getattr(self, f"log_{name}").exp()

    @property
    def poly_sigma(self) -> float | Tensor:
        """Current ``poly_sigma``, a tensor if learnable."""
        return self._value("poly_sigma")

    @property
    def window_sigma(self) -> float | Tensor | None:
        """Current ``window_sigma``, a tensor if learnable."""
        return self._value("window_sigma")

    @property
    def eps(self) -> float | Tensor:
        """Current ``eps``, a tensor if learnable."""
        return self._value("eps")

    @no_autocast
    def forward(
        self,
        frame0: Tensor,
        frame1: Tensor,
        initial_flow: Tensor | None = None,
        return_confidence: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Flow from ``frame0`` to ``frame1``; see :func:`farneback`."""
        return farneback(
            frame0,
            frame1,
            pyr_scale=self.pyr_scale,
            levels=self.levels,
            winsize=self.winsize,
            iterations=self.iterations,
            poly_n=self.poly_n,
            poly_sigma=self.poly_sigma,
            window=self.window,
            window_sigma=self.window_sigma,
            eps=self.eps,
            data_range=self.data_range,
            initial_flow=initial_flow,
            return_confidence=return_confidence,
        )

    def extra_repr(self) -> str:
        fields = [
            f"pyr_scale={self.pyr_scale}",
            f"levels={self.levels}",
            f"winsize={self.winsize}",
            f"iterations={self.iterations}",
            f"poly_n={self.poly_n}",
            f"window={self.window!r}",
        ]
        fields += [f"{name}={value}" for name, value in self._fixed.items() if value is not None]
        fields.append(f"data_range={self.data_range}")
        learnable = [name for name in _LEARNABLE if name not in self._fixed]
        if learnable:
            fields.append(f"learnable={learnable}")
        return ", ".join(fields)

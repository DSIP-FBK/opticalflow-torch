"""Internal building blocks shared by the flow methods (filters, resizing, pyramids, checks).

Several functions reproduce OpenCV operations exactly (same kernels, coefficients, borders and
rounding of sizes), so that ported algorithms can be compared with OpenCV number for number.
"""

from __future__ import annotations

import functools
import math
import operator
from collections.abc import Callable
from fractions import Fraction
from typing import Literal, NamedTuple, ParamSpec, TypeVar

import torch
import torch.nn.functional as F
from torch import Tensor

Border = Literal["reflect101", "replicate", "zeros"]
"""Border extrapolation: OpenCV's ``BORDER_REFLECT_101`` (``gfedcb|abcdefgh|gfedcba``),
``BORDER_REPLICATE`` (``aaaaaa|abcdefgh|hhhhhhh``) or zeros."""

_BORDERS = ("reflect101", "replicate", "zeros")

P = ParamSpec("P")
R = TypeVar("R")


# ----------------------------------------------------------------------------------------------
# Autocast
# ----------------------------------------------------------------------------------------------


def _autocast_devices() -> frozenset[str]:
    """Device types that :func:`torch.autocast` supports, among those PyTorch knows of."""
    if not hasattr(torch.amp, "is_autocast_available"):  # PyTorch 2.2 accepts these
        return frozenset(("cpu", "cuda", "xpu", "ipu", "hpu", "xla"))
    candidates = ("cpu", "cuda", "xpu", "mps", "hpu", "mtia", "xla", "ipu")
    return frozenset(t for t in candidates if torch.amp.is_autocast_available(t))


_AUTOCAST_DEVICES = _autocast_devices()  # once: torch.compile cannot trace is_autocast_available (PyTorch 2.7)


def no_autocast(fn: Callable[P, R]) -> Callable[P, R]:
    """Run ``fn`` with autocast disabled for the device type of its first argument (after ``self``), a tensor.

    Inside :func:`torch.autocast`, convolutions and some other operations run in float16 or
    bfloat16 whatever the dtype of their inputs: squared gradients of 0-255 frames overflow
    float16, and flows lose precision. With this, the public functions give the same results
    inside and outside autocast. They compute on the device of their first argument; device types
    without autocast need nothing. On a module's ``forward``, it also covers the learnable values
    computed from the parameters, such as ``exp`` (float32 under CUDA autocast).
    """
    index = 1 if fn.__code__.co_varnames[0] == "self" else 0
    first = fn.__code__.co_varnames[index]

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        tensor = args[index] if len(args) > index else kwargs.get(first)
        if isinstance(tensor, Tensor) and tensor.device.type in _AUTOCAST_DEVICES:
            with torch.autocast(tensor.device.type, enabled=False):
                return fn(*args, **kwargs)
        return fn(*args, **kwargs)

    return wrapper


# ----------------------------------------------------------------------------------------------
# Input checks
# ----------------------------------------------------------------------------------------------


def check_image(image: Tensor, name: str = "image") -> None:
    """Check that ``image`` is a floating-point tensor of shape ``(B, C, H, W)``."""
    if not isinstance(image, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(image).__name__}")
    if image.dim() != 4:
        raise ValueError(f"{name} must have shape (B, C, H, W), got shape {tuple(image.shape)}")
    if not image.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype, got {image.dtype}")


def check_frames(frame0: Tensor, frame1: Tensor) -> None:
    """Check a pair of frames: each as :func:`check_image`, with equal shape, dtype and device."""
    check_image(frame0, "frame0")
    check_image(frame1, "frame1")
    if frame0.shape != frame1.shape:
        raise ValueError(
            f"frame0 and frame1 must have the same shape, got {tuple(frame0.shape)} and {tuple(frame1.shape)}"
        )
    if frame0.dtype != frame1.dtype:
        raise TypeError(f"frame0 and frame1 must have the same dtype, got {frame0.dtype} and {frame1.dtype}")
    if frame0.device != frame1.device:
        raise ValueError(f"frame0 and frame1 must be on the same device, got {frame0.device} and {frame1.device}")


def check_flow(flow: Tensor, like: Tensor | None = None, name: str = "flow", like_name: str = "frames") -> None:
    """Check that ``flow`` is a floating-point ``(B, 2, H, W)`` tensor.

    Parameters
    ----------
    flow : Tensor
        Tensor to check.
    like : Tensor, optional
        Image of shape ``(B, C, H, W)`` whose batch size, spatial size, dtype and device the
        flow must share.
    name, like_name : str
        Names of ``flow`` and ``like`` in error messages.
    """
    if not isinstance(flow, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(flow).__name__}")
    if flow.dim() != 4 or flow.shape[1] != 2:
        raise ValueError(f"{name} must have shape (B, 2, H, W), got shape {tuple(flow.shape)}")
    if not flow.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype, got {flow.dtype}")
    if like is None:
        return
    b, _, h, w = like.shape
    if (flow.shape[0], flow.shape[2], flow.shape[3]) != (b, h, w):
        raise ValueError(f"{name} must have shape ({b}, 2, {h}, {w}) to match the {like_name}, got {tuple(flow.shape)}")
    if flow.dtype != like.dtype:
        raise TypeError(f"{name} must have the dtype of the {like_name} ({like.dtype}), got {flow.dtype}")
    if flow.device != like.device:
        raise ValueError(f"{name} must be on the device of the {like_name} ({like.device}), got {flow.device}")


def check_size(size: tuple[int, int], name: str = "size") -> tuple[int, int]:
    """Check an output ``(height, width)`` of positive integers and return it as a tuple."""
    try:
        height, width = (operator.index(v) for v in size)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a (height, width) pair of integers, got {size!r}") from None
    if height < 1 or width < 1:
        raise ValueError(f"{name} must contain positive integers, got {size!r}")
    return height, width


def as_scalar(value: float | Tensor, like: Tensor, name: str = "value") -> Tensor:
    """Return ``value`` as a 0-d tensor with the dtype and device of ``like``.

    A tensor keeps its autograd graph; a number becomes a constant.
    """
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must be a scalar, got a tensor of shape {tuple(value.shape)}")
        return value.reshape(()).to(dtype=like.dtype, device=like.device)
    if isinstance(value, (int, float)):
        return torch.full((), float(value), dtype=like.dtype, device=like.device)
    raise TypeError(f"{name} must be a float or a tensor, got {type(value).__name__}")


# ----------------------------------------------------------------------------------------------
# Kernels
# ----------------------------------------------------------------------------------------------


def cv_round(value: float) -> int:
    """OpenCV's ``cvRound``: nearest integer, ties to even."""
    return round(value)


def gaussian_kernel1d(
    sigma: float | Tensor,
    radius: int,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Normalised Gaussian kernel of length ``2 * radius + 1``, differentiable in ``sigma``.

    Parameters
    ----------
    sigma : float or Tensor
        Standard deviation(s), positive. A tensor of shape ``S`` gives kernels of shape
        ``(*S, 2 * radius + 1)``, e.g. one per channel for :func:`separable_filter`.
    radius : int
        Half-size of the kernel.
    dtype, device : optional
        Of the result; by default those of ``sigma`` (a tensor) or the default dtype on CPU.

    Returns
    -------
    Tensor
        Kernel(s) summing to 1 along the last dimension.
    """
    if not isinstance(radius, int) or radius < 0:
        raise ValueError(f"radius must be a non-negative integer, got {radius!r}")
    if not isinstance(sigma, Tensor):
        if not sigma > 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        sigma = torch.full((), float(sigma), dtype=dtype or torch.get_default_dtype(), device=device)
    elif dtype is not None or device is not None:
        sigma = sigma.to(dtype=dtype, device=device)
    x = torch.arange(-radius, radius + 1, dtype=sigma.dtype, device=sigma.device)
    w = torch.exp(-0.5 * (x / sigma.unsqueeze(-1)) ** 2)
    return w / w.sum(-1, keepdim=True)


# Kernels OpenCV uses for sigma <= 0 (getGaussianKernelBitExact).
_SMALL_GAUSSIAN_KERNELS = {
    1: (1.0,),
    3: (0.25, 0.5, 0.25),
    5: (0.0625, 0.25, 0.375, 0.25, 0.0625),
    7: (0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125),
    9: tuple(v / 256 for v in (4, 13, 30, 51, 60, 51, 30, 13, 4)),
}


def cv_gaussian_kernel(
    ksize: int,
    sigma: float,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> Tensor:
    """OpenCV's ``cv::getGaussianKernel(ksize, sigma)``, value for value.

    For ``sigma <= 0`` OpenCV uses fixed kernels for ``ksize`` 1, 3, 5, 7 and 9, and otherwise
    ``sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8``. Values are computed in double precision and
    rounded to ``dtype``, as OpenCV does for ``CV_32F`` kernels.

    Parameters
    ----------
    ksize : int
        Kernel length, positive and odd.
    sigma : float
        Standard deviation; ``<= 0`` derives it from ``ksize``.
    dtype, device : optional
        Of the result.

    Returns
    -------
    Tensor
        Kernel of shape ``(ksize,)``.
    """
    if not isinstance(ksize, int) or ksize < 1 or ksize % 2 == 0:
        raise ValueError(f"ksize must be a positive odd integer, got {ksize!r}")
    if sigma <= 0 and ksize in _SMALL_GAUSSIAN_KERNELS:
        values = list(_SMALL_GAUSSIAN_KERNELS[ksize])
    else:
        # Same operations and order as OpenCV's bit-exact version, in double precision; its
        # ksize * 0.15 + 0.35 is a fused multiply-add, i.e. rounded once.
        sigma_x = sigma if sigma > 0 else float(Fraction(ksize) * Fraction(0.15) + Fraction(0.35))
        scale2x = -0.125 / (sigma_x * sigma_x)
        tail = [math.exp((x * x) * scale2x) for x in range(1 - ksize, 0, 2)]  # x counts half-pixels
        inv = 1.0 / (sum(tail) * 2 + 1.0)
        tail = [t * inv for t in tail]
        values = [*tail, inv, *tail[::-1]]
    return torch.tensor(values, dtype=dtype, device=device)


# ----------------------------------------------------------------------------------------------
# Borders and separable filtering
# ----------------------------------------------------------------------------------------------


def _check_border(border: str) -> None:
    if border not in _BORDERS:
        raise ValueError(f"border must be one of {_BORDERS}, got {border!r}")


def _border_index(n: int, before: int, after: int, border: str, device: torch.device) -> Tensor:
    """Source index of each position of a padded axis, as ``cv::borderInterpolate``."""
    i = torch.arange(-before, n + after, device=device)
    if border == "replicate" or n == 1:
        return i.clamp(0, n - 1)
    period = 2 * (n - 1)  # reflect101 is periodic, which also covers pads longer than the axis
    i = torch.remainder(i, period)
    return torch.where(i >= n, period - i, i)


def pad_border(x: Tensor, padding: tuple[int, int, int, int], border: Border = "reflect101") -> Tensor:
    """Pad the last two dimensions of ``x``, of any size, like OpenCV's border modes.

    Parameters
    ----------
    x : Tensor
        Tensor of shape ``(..., H, W)``.
    padding : tuple of int
        ``(left, right, top, bottom)``, as in :func:`torch.nn.functional.pad`.
    border : {"reflect101", "replicate", "zeros"}
        See :data:`Border`. Pads may exceed the size of ``x``, as in OpenCV.

    Returns
    -------
    Tensor
        Tensor of shape ``(..., H + top + bottom, W + left + right)``.
    """
    _check_border(border)
    left, right, top, bottom = padding
    if min(padding) < 0:
        raise ValueError(f"padding must be non-negative, got {padding}")
    if border == "zeros":
        return F.pad(x, (left, right, top, bottom)) if any(padding) else x
    if left or right:
        x = x.index_select(-1, _border_index(x.shape[-1], left, right, border, x.device))
    if top or bottom:
        x = x.index_select(-2, _border_index(x.shape[-2], top, bottom, border, x.device))
    return x


def _depthwise_weight(kernel: Tensor, channels: int, like: Tensor, name: str) -> Tensor:
    if kernel.dim() not in (1, 2) or kernel.shape[-1] % 2 == 0:
        raise ValueError(f"{name} must have shape (k,) or (C, k) with k odd, got {tuple(kernel.shape)}")
    if kernel.dim() == 2 and kernel.shape[0] != channels:
        raise ValueError(f"{name} has {kernel.shape[0]} rows for an input with {channels} channels")
    kernel = kernel.to(dtype=like.dtype, device=like.device)
    return kernel.reshape(-1, 1, kernel.shape[-1]).expand(channels, 1, kernel.shape[-1])


def separable_filter(
    x: Tensor,
    kernel_x: Tensor | None,
    kernel_y: Tensor | None,
    border: Border = "reflect101",
) -> Tensor:
    """Correlate each channel with ``kernel_x`` along rows, then ``kernel_y`` along columns.

    The equivalent of ``cv2.sepFilter2D(x, -1, kernel_x, kernel_y, borderType=...)`` per channel,
    and of ``cv2.GaussianBlur`` with :func:`cv_gaussian_kernel` kernels. Differentiable with
    respect to ``x`` and the kernels. With two or more channels the convolutions are depthwise and
    run in full float32 precision on CUDA; with one channel cuDNN may compute the kernel gradient
    in TF32 unless ``torch.backends.cudnn.allow_tf32`` is False.

    Parameters
    ----------
    x : Tensor
        Input, shape ``(B, C, H, W)``.
    kernel_x, kernel_y : Tensor or None
        Kernels of odd length ``k``, shape ``(k,)`` for all channels or ``(C, k)`` per channel;
        ``None`` skips that direction. Cast to the dtype and device of ``x``.
    border : {"reflect101", "replicate", "zeros"}
        See :data:`Border`. ``"reflect101"`` is OpenCV's default.

    Returns
    -------
    Tensor
        Filtered input, shape ``(B, C, H, W)``.
    """
    check_image(x, "x")
    _check_border(border)
    channels = x.shape[1]
    if kernel_x is not None:
        weight = _depthwise_weight(kernel_x, channels, x, "kernel_x").unsqueeze(2)  # (C, 1, 1, k)
        r = weight.shape[-1] // 2
        x = F.conv2d(pad_border(x, (r, r, 0, 0), border), weight, groups=channels)
    if kernel_y is not None:
        weight = _depthwise_weight(kernel_y, channels, x, "kernel_y").unsqueeze(3)  # (C, 1, k, 1)
        r = weight.shape[-2] // 2
        x = F.conv2d(pad_border(x, (0, 0, r, r), border), weight, groups=channels)
    return x


# ----------------------------------------------------------------------------------------------
# Resizing (cv2.resize)
# ----------------------------------------------------------------------------------------------


def _table_dtype(device: torch.device) -> torch.dtype:
    return torch.float32 if device.type == "mps" else torch.float64


def _linear_taps(n_in: int, n_out: int, area: bool, like: Tensor) -> tuple[Tensor, Tensor]:
    """Two-tap indices and weights of OpenCV's INTER_LINEAR, or of INTER_AREA when enlarging."""
    inv_scale = n_out / n_in
    scale = 1.0 / inv_scale  # computed as OpenCV does
    d = torch.arange(n_out, dtype=_table_dtype(like.device), device=like.device)
    if area:
        start = torch.floor(d * scale)
        # rounded to float before the wrap, as in OpenCV: a fraction just below 1 becomes 0
        frac = ((d + 1) - (start + 1) * inv_scale).to(torch.float32)
        frac = torch.where(frac <= 0, 0.0, frac - frac.floor())
    else:
        coord = (d + 0.5) * scale - 0.5
        start = coord.floor()
        frac = torch.where(start < 0, 0.0, coord - start)
    start = start.long().clamp(min=0)
    frac = torch.where(start >= n_in - 1, 0.0, frac).to(like.dtype)  # only the weights are rounded
    start = start.clamp(max=n_in - 1)
    index = torch.stack((start, (start + 1).clamp(max=n_in - 1)), dim=-1)
    weight = torch.stack((1 - frac, frac), dim=-1)
    return index, weight


def _legacy_linear_taps(n_in: int, n_out: int, vertical: bool, like: Tensor) -> tuple[Tensor, Tensor]:
    """INTER_LINEAR taps of OpenCV 4, which OpenCV 5 keeps for inputs with a single row or column.

    Coordinates and weights are float32 for every depth, and the vertical weights are not reset
    at the edges (both taps then read the same clamped row).
    """
    d = torch.arange(n_out, dtype=_table_dtype(like.device), device=like.device)
    coord = ((d + 0.5) * (1.0 / (n_out / n_in)) - 0.5).to(torch.float32)
    start = coord.floor()
    frac = coord - start
    start = start.long()
    if not vertical:
        frac = torch.where((start < 0) | (start >= n_in - 1), 0.0, frac)
    index = torch.stack((start, start + 1), dim=-1).clamp(0, n_in - 1)
    weight = torch.stack((1 - frac, frac), dim=-1).to(like.dtype)
    return index, weight


def _area_taps(n_in: int, n_out: int, like: Tensor) -> tuple[Tensor, Tensor]:
    """Indices and weights of OpenCV's INTER_AREA when shrinking (pixel overlap fractions)."""
    scale = 1.0 / (n_out / n_in)
    taps = math.floor(scale) + 2  # a partial pixel on each side and the whole ones in between
    table_dtype = _table_dtype(like.device)
    d = torch.arange(n_out, dtype=table_dtype, device=like.device)
    start = d * scale
    stop = start + scale
    cell = (n_in - start).clamp(max=scale)
    last = stop.floor().clamp(max=n_in - 1)  # first pixel of the partial right overlap
    first = torch.minimum(start.ceil(), last)  # first whole pixel
    j = (first - 1).unsqueeze(-1) + torch.arange(taps, dtype=table_dtype, device=like.device)
    left = (first - start).unsqueeze(-1)
    right = (stop - last).unsqueeze(-1)
    first, last, cell = first.unsqueeze(-1), last.unsqueeze(-1), cell.unsqueeze(-1)
    # OpenCV drops partial overlaps up to 1e-3 pixels
    weight = torch.where((j >= first) & (j < last), 1.0, 0.0)
    weight = torch.where((j == first - 1) & (left > 1e-3), left, weight)
    weight = torch.where((j == last) & (right > 1e-3), torch.minimum(right.clamp(max=1.0), cell), weight)
    weight = (weight / cell).to(like.dtype)
    return j.clamp(0, n_in - 1).long(), weight


def _apply_taps(x: Tensor, dim: int, index: Tensor, weight: Tensor) -> Tensor:
    """Weighted sums of ``x`` along ``dim`` (-1 or -2) at ``index``, both shaped ``(n_out, taps)``."""
    n_out, taps = index.shape
    gathered = x.index_select(dim, index.reshape(-1)).unflatten(dim, (n_out, taps))
    if dim == -1:
        return (gathered * weight).sum(-1)
    return (gathered * weight.unsqueeze(-1)).sum(-2)


def resize(x: Tensor, size: tuple[int, int], interpolation: Literal["linear", "area"] = "linear") -> Tensor:
    """Resize the last two dimensions like ``cv2.resize`` for floating-point images.

    ``"linear"`` is ``INTER_LINEAR``: bilinear with half-pixel centres, edge pixels replicated,
    no antialiasing. ``"area"`` is ``INTER_AREA``: averages over pixel overlaps when shrinking,
    and OpenCV's two-tap variant when enlarging along either axis. Coefficients are computed as
    in OpenCV 5 (which keeps OpenCV 4's ``INTER_LINEAR`` for inputs with a single row or column),
    rows before columns, except that float64 inputs get double-precision ``"area"`` weights where
    OpenCV rounds them to float: up to about 2e-5 apart on 0-255 values. Returns ``x`` itself when
    the size does not change, as OpenCV copies. Differentiable with respect to ``x``.

    Parameters
    ----------
    x : Tensor
        Input, shape ``(..., H, W)``, floating point.
    size : tuple of int
        Output ``(height, width)``. Note that OpenCV's ``dsize`` is ``(width, height)``.
    interpolation : {"linear", "area"}
        OpenCV's ``INTER_LINEAR`` or ``INTER_AREA``.

    Returns
    -------
    Tensor
        Resized input, shape ``(..., height, width)``.
    """
    if not isinstance(x, Tensor) or not x.is_floating_point():
        raise TypeError("x must be a floating-point tensor")
    if x.dim() < 2:
        raise ValueError(f"x must have shape (..., H, W), got shape {tuple(x.shape)}")
    height, width = check_size(size)
    if interpolation not in ("linear", "area"):
        raise ValueError(f"interpolation must be 'linear' or 'area', got {interpolation!r}")
    h_in, w_in = x.shape[-2:]
    area = interpolation == "area"
    shrink = area and height <= h_in and width <= w_in  # OpenCV averages only when both shrink
    legacy = not area and (h_in == 1 or w_in == 1)
    for dim, n_in, n_out in ((-1, w_in, width), (-2, h_in, height)):
        if n_out == n_in:
            continue
        if shrink:
            taps = _area_taps(n_in, n_out, x)
        elif legacy:
            taps = _legacy_linear_taps(n_in, n_out, dim == -2, x)
        else:
            taps = _linear_taps(n_in, n_out, area, x)
        x = _apply_taps(x, dim, *taps)
    return x


# ----------------------------------------------------------------------------------------------
# Farneback's image pyramid
# ----------------------------------------------------------------------------------------------


class PyramidLevel(NamedTuple):
    """One level of :func:`pyramid_levels`."""

    scale: float
    """Size ratio to the full-resolution frame, ``pyr_scale ** k`` as OpenCV computes it."""
    height: int
    width: int


def pyramid_levels(height: int, width: int, pyr_scale: float, levels: int, min_size: int = 32) -> list[PyramidLevel]:
    """Levels of OpenCV's Farneback pyramid, finest (full resolution) first.

    As ``FarnebackOpticalFlowImpl::calc``: coarser levels whose unrounded width or height
    would be below ``min_size`` are dropped, and sizes are ``cvRound(size * scale)``.

    Parameters
    ----------
    height, width : int
        Size of the full-resolution frames.
    pyr_scale : float
        Size ratio between consecutive levels, in (0, 1).
    levels : int
        Number of coarser levels requested on top of the full-resolution one.
    min_size : int
        Smallest allowed width or height of a coarser level.

    Returns
    -------
    list of PyramidLevel
        ``1 + min(levels, levels allowed by min_size)`` levels.
    """
    if not 0 < pyr_scale < 1:
        raise ValueError(f"pyr_scale must be in (0, 1), got {pyr_scale}")
    if not isinstance(levels, int) or levels < 0:
        raise ValueError(f"levels must be a non-negative integer, got {levels!r}")
    count, scale = 0, 1.0
    while count < levels:
        scale *= pyr_scale
        if width * scale < min_size or height * scale < min_size:
            break
        count += 1
    result, scale = [], 1.0
    for _ in range(count + 1):
        result.append(PyramidLevel(scale, cv_round(height * scale), cv_round(width * scale)))
        scale *= pyr_scale
    return result


def pyramid_level(image: Tensor, level: PyramidLevel) -> Tensor:
    """One level of OpenCV's Farneback pyramid, made from the full-resolution ``image``.

    ``GaussianBlur`` with ``sigma = (1 / scale - 1) / 2`` and ``ksize = max(cvRound(5 * sigma) | 1, 3)``
    (so the full-resolution level gets OpenCV's 3-tap ``[1, 2, 1] / 4`` blur), reflect-101
    border, then :func:`resize` with ``"linear"``.

    Parameters
    ----------
    image : Tensor
        Full-resolution frames, shape ``(B, C, H, W)``.
    level : PyramidLevel
        From :func:`pyramid_levels`.

    Returns
    -------
    Tensor
        Level of shape ``(B, C, level.height, level.width)``.
    """
    sigma = (1.0 / level.scale - 1) * 0.5
    ksize = max(cv_round(sigma * 5) | 1, 3)
    kernel = cv_gaussian_kernel(ksize, sigma, dtype=image.dtype, device=image.device)
    blurred = separable_filter(image, kernel, kernel, border="reflect101")
    return resize(blurred, (level.height, level.width), "linear")


def build_pyramid(image: Tensor, pyr_scale: float, levels: int, min_size: int = 32) -> list[Tensor]:
    """OpenCV's Farneback pyramid of ``image``, finest first; see :func:`pyramid_levels`."""
    check_image(image)
    sizes = pyramid_levels(image.shape[-2], image.shape[-1], pyr_scale, levels, min_size)
    return [pyramid_level(image, level) for level in sizes]

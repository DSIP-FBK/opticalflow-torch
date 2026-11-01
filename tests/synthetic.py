"""Synthetic frames with known flow, and conversions to and from OpenCV's array layouts.

Flows follow the library convention ``frame0(p) = frame1(p + flow(p))``, with ``p = (x, y)`` in
pixels, x along columns.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor


def smooth_texture(
    height: int,
    width: int,
    *,
    batch: int = 1,
    channels: int = 1,
    sigma: float = 2.0,
    seed: int = 0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Gaussian-filtered white noise scaled to [0, 255] per image, shape ``(B, C, H, W)``.

    The filtering is circular, so the texture is periodic and ``torch.roll`` gives exact
    integer translations. ``sigma`` is the filter's standard deviation in pixels.
    """
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(batch, channels, height, width, generator=generator, dtype=torch.float64)
    fy = torch.fft.fftfreq(height, dtype=torch.float64).unsqueeze(-1)
    fx = torch.fft.rfftfreq(width, dtype=torch.float64)
    gain = torch.exp(-2 * (math.pi * sigma) ** 2 * (fx**2 + fy**2))
    image = torch.fft.irfft2(torch.fft.rfft2(noise) * gain, s=(height, width))
    low = image.amin(dim=(-2, -1), keepdim=True)
    high = image.amax(dim=(-2, -1), keepdim=True)
    return (255 * (image - low) / (high - low)).to(device=device, dtype=dtype)


class AnalyticTexture:
    """Smooth random texture defined at any real coordinates, values in (0, 255).

    A sum of plane waves with random directions and phases and log-uniform wavelengths,
    passed through ``127.5 + 127.5 * tanh(0.75 * s)`` with ``s`` of unit variance.

    Parameters
    ----------
    seed : int
        Seed of the random waves.
    channels : int
        Number of independent channels.
    waves : int
        Number of plane waves per channel.
    min_wavelength, max_wavelength : float
        Range of the wavelengths, in pixels.
    """

    def __init__(
        self,
        seed: int = 0,
        *,
        channels: int = 1,
        waves: int = 24,
        min_wavelength: float = 8.0,
        max_wavelength: float = 64.0,
    ) -> None:
        generator = torch.Generator().manual_seed(seed)
        shape = (channels, waves)
        angle = 2 * math.pi * torch.rand(shape, generator=generator, dtype=torch.float64)
        log_range = math.log(max_wavelength / min_wavelength)
        wavelength = min_wavelength * torch.exp(log_range * torch.rand(shape, generator=generator, dtype=torch.float64))
        self.kx = 2 * math.pi * torch.cos(angle) / wavelength
        self.ky = 2 * math.pi * torch.sin(angle) / wavelength
        self.phase = 2 * math.pi * torch.rand(shape, generator=generator, dtype=torch.float64)
        self.channels, self.waves = channels, waves

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        """Values at columns ``x`` and rows ``y`` (broadcastable), shape ``(*shape, C)``, float64."""
        x, y = torch.broadcast_tensors(x.double().cpu(), y.double().cpu())
        x, y = x.unsqueeze(-1), y.unsqueeze(-1)
        total = torch.zeros(*x.shape[:-1], self.channels, dtype=torch.float64)
        for i in range(self.waves):
            total += torch.cos(self.kx[:, i] * x + self.ky[:, i] * y + self.phase[:, i])
        return 127.5 + 127.5 * torch.tanh(0.75 * total / math.sqrt(self.waves / 2))

    def image(
        self,
        height: int,
        width: int,
        flow: Tensor | None = None,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """Texture at the pixels displaced by ``flow``: ``out[b, :, y, x] = T(x + flow_x, y + flow_y)``.

        Returns shape ``(B, C, H, W)``, with ``B`` the batch size of ``flow`` (1 without flow).
        """
        x = torch.arange(width, dtype=torch.float64).expand(1, height, width)
        y = torch.arange(height, dtype=torch.float64).unsqueeze(-1).expand(1, height, width)
        if flow is not None:
            flow = flow.detach().double().cpu()
            x, y = x + flow[:, 0], y + flow[:, 1]
        return self(x, y).permute(0, 3, 1, 2).to(device=device, dtype=dtype).contiguous()


def _grid(height: int, width: int, center: tuple[float, float] | None) -> tuple[Tensor, Tensor, float, float]:
    cx, cy = ((width - 1) / 2, (height - 1) / 2) if center is None else center
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float64), torch.arange(width, dtype=torch.float64), indexing="ij"
    )
    return x, y, cx, cy


def _as_flow(fx: Tensor, fy: Tensor, batch: int, device: torch.device | str, dtype: torch.dtype) -> Tensor:
    flow = torch.stack((fx, fy)).unsqueeze(0).expand(batch, -1, -1, -1)
    return flow.to(device=device, dtype=dtype).contiguous()


def translation_flow(
    height: int,
    width: int,
    dx: float,
    dy: float,
    *,
    batch: int = 1,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Constant flow ``(dx, dy)``, shape ``(B, 2, H, W)``."""
    fx = torch.full((height, width), float(dx), dtype=torch.float64)
    return _as_flow(fx, torch.full_like(fx, float(dy)), batch, device, dtype)


def rotation_flow(
    height: int,
    width: int,
    angle: float,
    *,
    center: tuple[float, float] | None = None,
    batch: int = 1,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Flow ``R (p - c) + c - p`` of a rotation by ``angle`` degrees about ``center`` (x, y).

    ``center`` defaults to the middle of the image.
    """
    x, y, cx, cy = _grid(height, width, center)
    cos, sin = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    fx = cos * (x - cx) - sin * (y - cy) + cx - x
    fy = sin * (x - cx) + cos * (y - cy) + cy - y
    return _as_flow(fx, fy, batch, device, dtype)


def zoom_flow(
    height: int,
    width: int,
    factor: float,
    *,
    center: tuple[float, float] | None = None,
    batch: int = 1,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Flow ``(factor - 1) (p - c)`` of a zoom (divergence) about ``center`` (x, y).

    ``center`` defaults to the middle of the image.
    """
    x, y, cx, cy = _grid(height, width, center)
    return _as_flow((factor - 1) * (x - cx), (factor - 1) * (y - cy), batch, device, dtype)


def make_pair(
    texture: AnalyticTexture,
    flow: Tensor,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[Tensor, Tensor]:
    """Frames with exactly ``frame0(p) = frame1(p + flow(p))``, from ``texture``.

    ``frame1`` samples the texture at the pixels and ``frame0`` at the displaced pixels.
    Both have the batch size of ``flow`` and, by default, its device and dtype.
    """
    device = flow.device if device is None else device
    dtype = flow.dtype if dtype is None else dtype
    height, width = flow.shape[-2:]
    frame0 = texture.image(height, width, flow, device=device, dtype=dtype)
    frame1 = texture.image(height, width, device=device, dtype=dtype).expand_as(frame0).contiguous()
    return frame0, frame1


def image_to_cv(image: Tensor, index: int = 0) -> np.ndarray:
    """Image ``index`` of a ``(B, C, H, W)`` tensor as a float32 ``(H, W)`` or ``(H, W, C)`` array."""
    array = image[index].detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
    return np.ascontiguousarray(array[..., 0] if array.shape[-1] == 1 else array)


def image_from_cv(
    array: np.ndarray, *, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32
) -> Tensor:
    """An ``(H, W)`` or ``(H, W, C)`` array as a ``(1, C, H, W)`` tensor."""
    tensor = torch.from_numpy(np.asarray(array, dtype=np.float64))
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(-1)
    return tensor.permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype).contiguous()


def flow_to_cv(flow: Tensor, index: int = 0) -> np.ndarray:
    """Flow ``index`` of a ``(B, 2, H, W)`` tensor as OpenCV's float32 ``(H, W, 2)`` array."""
    return np.ascontiguousarray(flow[index].detach().cpu().permute(1, 2, 0).numpy().astype(np.float32))


def flow_from_cv(
    array: np.ndarray, *, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32
) -> Tensor:
    """OpenCV's ``(H, W, 2)`` flow array as a ``(1, 2, H, W)`` tensor."""
    return image_from_cv(array, device=device, dtype=dtype)

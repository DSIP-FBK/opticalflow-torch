"""Differentiable optical flow in pure PyTorch.

Frames are ``(B, C, H, W)`` float tensors. Flows are ``(B, 2, H, W)`` displacements in
pixels (channel 0 is x, channel 1 is y) such that
``frame0[y, x] ≈ frame1[y + flow_y, x + flow_x]``, the OpenCV convention.
"""

from importlib.metadata import PackageNotFoundError, version

from .farneback import Farneback, farneback
from .interpolate import interpolate_frames
from .lucas_kanade import LucasKanade, lucas_kanade
from .metrics import consistency, endpoint_error, photometric_error
from .visualize import flow_to_rgb
from .warp import resize_flow, warp

try:
    __version__ = version("opticalflow-torch")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.0.0"

__all__ = [
    "Farneback",
    "LucasKanade",
    "consistency",
    "endpoint_error",
    "farneback",
    "flow_to_rgb",
    "interpolate_frames",
    "lucas_kanade",
    "photometric_error",
    "resize_flow",
    "warp",
]

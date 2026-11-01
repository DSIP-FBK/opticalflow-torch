"""One entry point for both flow methods, and interpolation that computes the flow itself."""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from typing import Any, Literal

from torch import Tensor, nn

from .farneback import Farneback, _check_int, farneback
from .interpolate import interpolate_frames
from .lucas_kanade import LucasKanade, lucas_kanade

_METHODS = ("farneback", "lucas_kanade")
_FUNCTIONS = {"farneback": farneback, "lucas_kanade": lucas_kanade}
_MODULES = {"farneback": Farneback, "lucas_kanade": LucasKanade}


def _check_method(method: str) -> str:
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    return method


def _interpolate(
    flow: Callable[[Tensor, Tensor], Tensor],
    frame0: Tensor,
    frame1: Tensor,
    steps: int | None,
    t: float | Sequence[float] | Tensor | None,
    bidirectional: bool,
    **kwargs: Any,
) -> Tensor:
    """:func:`interpolate_frames` with the flows that ``flow`` computes, at ``t`` or at ``k / (steps + 1)``."""
    if (steps is None) == (t is None):
        raise ValueError("exactly one of steps and t must be given")
    if steps is not None:
        steps = _check_int(steps, "steps", 1)
        t = [k / (steps + 1) for k in range(1, steps + 1)]
    backward_flow = flow(frame1, frame0) if bidirectional else None
    return interpolate_frames(frame0, frame1, flow(frame0, frame1), t, backward_flow=backward_flow, **kwargs)


def optical_flow(
    frame0: Tensor, frame1: Tensor, *, method: Literal["farneback", "lucas_kanade"] = "farneback", **options: Any
) -> Tensor | tuple[Tensor, Tensor]:
    """Flow from ``frame0`` to ``frame1`` with :func:`farneback` or :func:`lucas_kanade`.

    Parameters
    ----------
    frame0, frame1 : Tensor
        Frames, shape ``(B, C, H, W)``, floating point.
    method : {"farneback", "lucas_kanade"}
        Flow method.
    **options
        Keywords of :func:`farneback` or :func:`lucas_kanade`, e.g. ``data_range``.

    Returns
    -------
    Tensor or tuple of Tensor
        What the method returns: the flow, shape ``(B, 2, H, W)``, and the confidence if requested.
    """
    return _FUNCTIONS[_check_method(method)](frame0, frame1, **options)


def interpolate(
    frame0: Tensor,
    frame1: Tensor,
    *,
    steps: int | None = None,
    t: float | Sequence[float] | Tensor | None = None,
    method: Literal["farneback", "lucas_kanade"] = "farneback",
    bidirectional: bool = False,
    match_distribution: bool = False,
    quantiles: int = 256,
    **options: Any,
) -> Tensor:
    """Frames between ``frame0`` and ``frame1``, along the flow computed between them.

    Computes the flow with :func:`optical_flow`, then the frames with :func:`interpolate_frames`,
    which takes a flow you already have. Give exactly one of ``steps`` and ``t``.

    Parameters
    ----------
    frame0, frame1 : Tensor
        Frames, shape ``(B, C, H, W)``, at times 0 and 1.
    steps : int, optional
        Number of frames in between, ``frame0`` and ``frame1`` not included (unlike
        ``torch.linspace``), at times ``k / (steps + 1)`` for ``k = 1, ..., steps``; 2 turns
        15-minute frames into 5-minute ones.
    t : float, sequence of float or Tensor, optional
        Times in [0, 1], as in :func:`interpolate_frames`.
    method : {"farneback", "lucas_kanade"}
        Flow method.
    bidirectional : bool
        Also compute the flow from ``frame1`` to ``frame0``, the ``backward_flow`` of
        :func:`interpolate_frames`.
    match_distribution : bool
        Probability matching; see :func:`interpolate_frames`.
    quantiles : int
        See :func:`interpolate_frames`.
    **options
        Keywords of :func:`farneback` or :func:`lucas_kanade`, except ``initial_flow`` and
        ``return_confidence``.

    Returns
    -------
    Tensor
        Frames of shape ``(B, T, C, H, W)``, one per time.
    """
    for name in ("initial_flow", "return_confidence"):
        if name in options:
            raise TypeError(f"interpolate does not take {name}; use optical_flow and interpolate_frames")
    flow = functools.partial(_FUNCTIONS[_check_method(method)], **options)
    return _interpolate(
        flow, frame0, frame1, steps, t, bidirectional, match_distribution=match_distribution, quantiles=quantiles
    )


class OpticalFlow(nn.Module):
    """Module form of :func:`optical_flow`, with optionally learnable parameters.

    Calling it gives the flow; :meth:`interpolate` gives the frames in between, as
    :func:`~opticalflow_torch.highlevel.interpolate` does.

    Parameters
    ----------
    method : {"farneback", "lucas_kanade"}
        Flow method, kept as the ``method`` attribute.
    **options
        Keywords of :class:`Farneback` or :class:`LucasKanade`, e.g. ``data_range`` or ``learnable``.
        That module is the ``estimator`` attribute; its parameters are this module's.
    """

    def __init__(self, method: Literal["farneback", "lucas_kanade"] = "farneback", **options: Any) -> None:
        super().__init__()
        self.method = _check_method(method)
        self.estimator = _MODULES[method](**options)

    def forward(
        self, frame0: Tensor, frame1: Tensor, initial_flow: Tensor | None = None, return_confidence: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Flow from ``frame0`` to ``frame1``; see :func:`farneback` or :func:`lucas_kanade`."""
        return self.estimator(frame0, frame1, initial_flow, return_confidence)

    def interpolate(
        self,
        frame0: Tensor,
        frame1: Tensor,
        *,
        steps: int | None = None,
        t: float | Sequence[float] | Tensor | None = None,
        bidirectional: bool = False,
        match_distribution: bool = False,
        quantiles: int = 256,
    ) -> Tensor:
        """Frames between ``frame0`` and ``frame1`` along this module's flow.

        See :func:`~opticalflow_torch.highlevel.interpolate`.
        """
        return _interpolate(
            self, frame0, frame1, steps, t, bidirectional, match_distribution=match_distribution, quantiles=quantiles
        )

    def extra_repr(self) -> str:
        return f"method={self.method!r}"

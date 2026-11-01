# Interpolation

[`interpolate_frames`][opticalflow_torch.interpolate.interpolate_frames] makes the frames between
`frame0` (time 0) and `frame1` (time 1) by moving both along the flow, instead of blending them in
place.

## Advection correction

Radar images come every few minutes. Rain summed from them shows stripes or separate blobs where
storms move more than their own size between two images. Interpolating intermediate frames along
the motion before summing removes this; for radar it is called advection correction (e.g.
Anagnostou and Krajewski, 1999)[^1].

At time \(t\), with \(f\) the flow from `frame0` to `frame1`, each output frame is

\[
I_t(\mathbf{x}) = (1 - t)\, \text{frame0}\bigl(\mathbf{x} - t\, f(\mathbf{x})\bigr)
+ t\, \text{frame1}\bigl(\mathbf{x} + (1 - t)\, f(\mathbf{x})\bigr),
\]

that is, `frame0` moved forward by \(t f\) and `frame1` moved back by \((1 - t) f\), weighted by
\(1 - t\) and \(t\). Taking the flow at the output pixel rather than following it along the path
is accurate when the flow varies little over the distance moved. With `backward_flow` \(b\), the
flow from `frame1` to `frame0`, the second term uses \(\text{frame1}(\mathbf{x} - (1 - t)\, b(\mathbf{x}))\),
so each frame moves with its own motion. Times 0 and 1 return `frame0` and `frame1` exactly.

The result has shape `(B, T, C, H, W)`, one frame per time.

## Probability matching

Blending smooths the field: bilinear sampling and any error in the flow lower the peaks and
spread the light rain, so the blend's distribution of values is narrower than the frames'. With
`match_distribution=True`, the values of each blend are remapped, per image and channel, so that
its quantiles become those interpolated between `frame0` and `frame1`,
\(Q_t = (1 - t)\, Q_0 + t\, Q_1\). Each value \(v\) goes through the blend's distribution function
\(F\), piecewise linear between `quantiles` quantiles, and then through \(Q_t\):
\(v \mapsto Q_t(F(v))\). The map is non-decreasing, so it keeps the spatial pattern of the blend
and only changes its values.

Where values of the blend tie, as the zeros of dry areas do, \(F\) jumps, and an exact matching
would jump too: tied pixels share one value, but separated by any amount they take different
quantiles. So \(F\) is averaged over \([v - h, v + h]\), with \(h\) a quarter of the mean spacing
of the target quantiles, which makes the map continuous, like the rest of the computation. Below
the blend's minimum, \(F\) is extended as an odd function, so the minimum keeps probability 0 and
maps to the lowest target value: dry pixels stay dry, even when the blend is drier than the frames,
as when rain enters through the border. Other tied values map to about the middle of their
targets, and the maximum to a little below the highest target. With `quantiles` equal to the
number of pixels, the result is close to rank matching.

## Example

Three Gaussian "rain cells" move 8 pixels right and 4 down between two frames. The flow is
estimated with [`farneback`][opticalflow_torch.farneback.farneback], then ten frames are made
between the two and averaged, as when accumulating rain over the interval.

```python
import torch

from opticalflow_torch import farneback, interpolate_frames, warp

size = 128
coords = torch.arange(float(size))
y, x = torch.meshgrid(coords, coords, indexing="ij")
cells = [(40, 40, 5), (64, 80, 8), (90, 50, 4)]  # row, column, radius in pixels
field = sum(torch.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * r**2)) for cy, cx, r in cells)
field = (255 * field / field.max()).view(1, 1, size, size)
motion = torch.tensor([8.0, 4.0]).view(1, 2, 1, 1).expand(1, 2, size, size)


def truth(t: float) -> torch.Tensor:
    """The field at time t: moved by t * motion."""
    return warp(field, -t * motion)


frame0, frame1 = truth(0.0), truth(1.0)
flow = farneback(frame0, frame1)

t = torch.arange(10) / 10  # 0, 0.1, ..., 0.9; time 1 starts the next interval
frames = interpolate_frames(frame0, frame1, flow, t)  # (1, 10, 1, 128, 128)
matched = interpolate_frames(frame0, frame1, flow, t, match_distribution=True)
blends = torch.stack([(1 - s) * frame0 + s * frame1 for s in t], dim=1)  # no motion
truths = torch.stack([truth(float(s)) for s in t], dim=1)

for name, guess in [("blend", blends), ("motion-compensated", frames), ("probability-matched", matched)]:
    error = (guess.mean(dim=1) - truths.mean(dim=1)).abs().mean()
    peak = guess[:, 5].max()  # at t = 0.5
    print(f"{name:20s} accumulation error {error:5.2f}  peak at t=0.5 {peak:6.1f}")
print(f"{'truth':20s} {'':24s}  peak at t=0.5 {truths[:, 5].max():6.1f}")
```

```text
blend                accumulation error  1.60  peak at t=0.5  218.1
motion-compensated   accumulation error  0.06  peak at t=0.5  255.0
probability-matched  accumulation error  0.10  peak at t=0.5  254.9
truth                                          peak at t=0.5  255.0
```

The plain blend shows each cell twice at \(t = 0.5\), with lower peaks, and its average over the
interval leaves two blobs per cell where the true accumulation is a streak. The motion-compensated
frames follow the cells. The flow is nearly exact on this synthetic motion, so probability matching
has little to correct here; it matters when errors in the flow, or cells that change, make the
blend smoother than the frames, as with real radar data.

[^1]: E. N. Anagnostou and W. F. Krajewski, "Real-time radar rainfall estimation. Part I:
    Algorithm formulation", Journal of Atmospheric and Oceanic Technology, 16, 189-197, 1999.

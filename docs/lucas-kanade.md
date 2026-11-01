# Lucas-Kanade

[`lucas_kanade`][opticalflow_torch.lucas_kanade.lucas_kanade] computes a dense Lucas-Kanade flow:
one vector per pixel, each from the window around that pixel, refined iteratively from coarse to
fine pyramid levels. The [`LucasKanade`][opticalflow_torch.lucas_kanade.LucasKanade] module wraps
it with optionally learnable parameters.

## The method

Lucas-Kanade fits a displacement to the window around each pixel. At each iteration, `frame1` is
warped back with the current flow, and at every pixel the update \(\boldsymbol{\delta}\) solves the
damped, linearised least-squares problem over the window,

\[
(G + \mu\, \mathrm{Id})\, \boldsymbol{\delta} = -\mathbf{b}, \qquad
G = \langle \nabla f_0\, \nabla f_0^\top \rangle_w, \qquad
\mathbf{b} = \bigl\langle \nabla f_0\, \bigl(\tilde f_1 - f_0\bigr) \bigr\rangle_w,
\]

where \(f_0\) is `frame0`, \(\tilde f_1\) is `frame1` warped by the current flow, \(\nabla f_0\) is
the gradient of `frame0` (central differences) and \(\langle\cdot\rangle_w\) is the average over the
window: a Gaussian by default, or a `winsize` box. \(G\) is the structure tensor. Pixels whose
displaced position falls outside `frame1` lose their weight in the averages, linearly over one
pixel.

The damping is `eps` plus a sixteenth of the squared residual that no displacement explains,

\[
\mu = \text{eps} + \tfrac{1}{16} \max\Bigl(0,\; \bigl\langle (\tilde f_1 - f_0)^2 \bigr\rangle_w
- \mathbf{b}^\top (G + \text{eps}\, \mathrm{Id})^{-1}\, \mathbf{b}\Bigr).
\]

Where a displacement accounts for the difference between the frames, the second term is close to
zero and `eps` alone damps the update. Where `frame0` is flat but `frame1` is not, as at the edge of
a rain area, it is about the whole squared residual, which keeps the update below about 2 pixels of
the pyramid level and its gradients moderate.

Classic Lucas-Kanade assumes a constant flow in each window. Here the flow is also averaged over
the window before each iteration, which keeps affine motions such as rotations and zooms, and
stops neighbouring estimates from drifting apart.

The update is repeated `iterations` times at each pyramid level, and the flow of a level, resized
and rescaled, starts the next finer one. The pyramid is built as that of
[`farneback`][opticalflow_torch.farneback.farneback], with the same `levels` and `pyr_scale`:
`levels` counts the levels added on top of the full resolution, and levels under 32 pixels are
skipped. Its coarser levels are then blurred once more by \([1, 2, 1] / 4\) along each axis, as the
full-resolution level is: Farneback's polynomial fits tolerate the fine detail left by the
resampling, differences do not.

Where the window has little texture, the structure tensor is close to singular and the damping
holds the update back: the flow there comes from the coarser levels and from the averaging with
neighbouring pixels. The structure tensor scales with the square of the intensities, and so should
`eps` (see [Value range](conventions.md#value-range)). With `return_confidence=True` you also get
the smallest eigenvalue of the structure tensor (without `eps`) at full resolution: near zero on
flat areas, on straight edges and where the displaced positions leave `frame1`, large on corners
and texture.

## Compared with OpenCV's `calcOpticalFlowPyrLK`

`cv2.calcOpticalFlowPyrLK` is sparse: it tracks a given set of points, typically corners from
`cv2.goodFeaturesToTrack`, and returns their new positions with a status and an error per point.

| | `cv2.calcOpticalFlowPyrLK` | `lucas_kanade` |
| --- | --- | --- |
| Output | new positions of the given points | a vector at every pixel, `(B, 2, H, W)` |
| Window | `winSize=(21, 21)`, uniform weights | `winsize=21`, Gaussian weights by default, `window="box"` for uniform ones |
| Image gradient | Scharr derivatives of `prevImg` | central differences of `frame0` |
| Pyramid | `maxLevel=3` levels added by halving | `levels=3` levels added, size ratio `pyr_scale` |
| Iterations | up to 30 per level, stopping when the update is below 0.01 pixels | a fixed number per level, `iterations=5` |
| Weak texture | points below `minEigThreshold` are marked as lost | `eps` and the unexplained residual damp the update; the confidence output reports the smallest eigenvalue |
| Gradients | none | with respect to the frames, `initial_flow`, `window_sigma` and `eps` |

The fixed number of iterations and the damping instead of a hard rejection keep the computation
differentiable and free of control flow that depends on the data. To reject weak estimates as
OpenCV does, threshold the confidence yourself, outside the part you differentiate.

## Compared with pysteps

pysteps' `dense_lucaskanade` (`pysteps.motion.lucaskanade`) detects Shi-Tomasi corners in the
precipitation field, tracks them with `cv2.calcOpticalFlowPyrLK`, removes outliers and
interpolates the sparse vectors onto the grid with radial basis functions. The result is a smooth,
large-scale motion field that extends over dry areas, set by a limited number of tracked features.

`lucas_kanade` instead estimates every pixel from its own window, so it resolves more local
detail, and it has no sparse vectors to interpolate: in dry areas the flow comes from the coarser
levels and from neighbouring pixels, and far from any rain it stays near zero (or near
`initial_flow`). For a field closer to pysteps', use a larger `winsize` and more `levels`, or
smooth the flow afterwards. `flow[0].numpy()` has the layout of a pysteps motion field: shape
`(2, H, W)`, forward displacements in pixels per time step, x component first.

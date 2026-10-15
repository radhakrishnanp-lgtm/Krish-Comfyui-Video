"""Pure-torch frame/size conforming logic (no ComfyUI imports, so it is unit-testable).

Images are ComfyUI IMAGE tensors: float [N, H, W, C].
"""
from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F


def next_valid_frames(n: int, stride: int = 8) -> int:
    """Smallest count >= n of the form stride*k + 1 (never rounds down)."""
    if n <= 1:
        return 1
    return ((n - 1 + stride - 1) // stride) * stride + 1


def next_multiple(v: int, m: int) -> int:
    return ((v + m - 1) // m) * m


@dataclass
class ConformInfo:
    orig_frames: int
    padded_frames: int
    orig_h: int
    orig_w: int
    padded_h: int
    padded_w: int
    pad_top: int
    pad_left: int
    frame_stride: int
    size_multiple: int

    def to_dict(self):
        return asdict(self)


def _frame_pad_indices(n: int, target: int, mode: str):
    """Source indices for the padded sequence; first n are identity."""
    idx = list(range(n))
    extra = target - n
    if extra <= 0:
        return idx
    if mode == "repeat_last":
        idx += [n - 1] * extra
    elif mode == "pingpong":
        # n-2, n-3, ... 0, 1, ... (bounces without duplicating the edge frame)
        if n == 1:
            idx += [0] * extra
        else:
            period = list(range(n - 2, -1, -1)) + list(range(1, n))
            idx += [period[i % len(period)] for i in range(extra)]
    else:
        raise ValueError(f"unknown frame pad mode: {mode}")
    return idx


def _spatial_pad(images, pad_t, pad_b, pad_l, pad_r, mode: str):
    if not (pad_t or pad_b or pad_l or pad_r):
        return images
    x = images.permute(0, 3, 1, 2)  # NCHW
    h, w = x.shape[2], x.shape[3]
    if mode == "reflect" and max(pad_t, pad_b) < h and max(pad_l, pad_r) < w:
        x = F.pad(x, (pad_l, pad_r, pad_t, pad_b), mode="reflect")
    else:  # replicate is always safe
        x = F.pad(x, (pad_l, pad_r, pad_t, pad_b), mode="replicate")
    return x.permute(0, 2, 3, 1).contiguous()


def prepare(images, frame_stride=8, size_multiple=32,
            frame_pad="repeat_last", spatial_pad="reflect", spatial_anchor="center"):
    """Pad (never resample) to LTX's grid. Original pixels stay bit-exact."""
    n, h, w, _ = images.shape
    tf = next_valid_frames(n, frame_stride)
    th, tw = next_multiple(h, size_multiple), next_multiple(w, size_multiple)

    idx = torch.tensor(_frame_pad_indices(n, tf, frame_pad), dtype=torch.long)
    out = images.index_select(0, idx.to(images.device))

    ph, pw = th - h, tw - w
    if spatial_anchor == "center":
        pt, pl = ph // 2, pw // 2
    else:  # top_left: content stays at origin
        pt, pl = 0, 0
    out = _spatial_pad(out, pt, ph - pt, pl, pw - pl, spatial_pad)

    return out, ConformInfo(n, tf, h, w, th, tw, pt, pl, frame_stride, size_multiple)


def _retime(frames, target: int, method: str):
    """Map k generated frames onto `target` frames spanning the same timeline."""
    k = frames.shape[0]
    if k == target:
        return frames
    if target == 1:
        return frames[:1]
    pos = torch.linspace(0, k - 1, target, device=frames.device)
    if method == "nearest":
        return frames.index_select(0, pos.round().long())
    lo = pos.floor().long()
    hi = pos.ceil().long().clamp(max=k - 1)
    t = (pos - lo.to(pos.dtype)).view(-1, 1, 1, 1).to(frames.dtype)
    return frames.index_select(0, lo) * (1 - t) + frames.index_select(0, hi) * t


def restore(generated, info, retime="nearest", resize_mode="bilinear",
            source=None, mask=None):
    """Trim padding and force exactly the original frame count and size.

    Returns (images, report) where report lists any correction applied.
    """
    i = info
    report = []
    g = generated

    if g.shape[0] != i.padded_frames:
        report.append(f"frames {g.shape[0]} -> {i.padded_frames} (retime={retime})")
        g = _retime(g, i.padded_frames, retime)

    if (g.shape[1], g.shape[2]) != (i.padded_h, i.padded_w):
        report.append(f"size {g.shape[2]}x{g.shape[1]} -> {i.padded_w}x{i.padded_h} (resize)")
        x = g.permute(0, 3, 1, 2)
        kw = {} if resize_mode == "nearest" else {"align_corners": False}
        x = F.interpolate(x, size=(i.padded_h, i.padded_w), mode=resize_mode, **kw)
        g = x.permute(0, 2, 3, 1).contiguous()

    g = g[: i.orig_frames,
          i.pad_top: i.pad_top + i.orig_h,
          i.pad_left: i.pad_left + i.orig_w]

    if source is not None and mask is not None:
        if source.shape != g.shape:
            raise ValueError(f"source {tuple(source.shape)} != output {tuple(g.shape)}")
        m = mask
        if m.dim() == 2:
            m = m.unsqueeze(0)
        if m.shape[1:] != g.shape[1:3]:
            m = F.interpolate(m.unsqueeze(1), size=g.shape[1:3], mode="bilinear",
                              align_corners=False).squeeze(1)
        if m.shape[0] not in (1, g.shape[0]):
            m = _retime(m.unsqueeze(-1), g.shape[0], "nearest").squeeze(-1)
        m = m.clamp(0, 1).unsqueeze(-1).to(g.dtype)
        g = source * (1 - m) + g * m
        report.append("composited over source with mask")

    assert g.shape[0] == i.orig_frames and g.shape[1:3] == (i.orig_h, i.orig_w)
    return g, report


def parity(a, b, atol=None):
    """Return list of mismatch strings between two IMAGE batches (empty == match)."""
    problems = []
    if a.shape[0] != b.shape[0]:
        problems.append(f"frame count {a.shape[0]} vs {b.shape[0]}")
    if a.shape[1:3] != b.shape[1:3]:
        problems.append(f"size {a.shape[2]}x{a.shape[1]} vs {b.shape[2]}x{b.shape[1]}")
    if a.shape[3] != b.shape[3]:
        problems.append(f"channels {a.shape[3]} vs {b.shape[3]}")
    return problems

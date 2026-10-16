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


def _waveform(audio):
    wav = audio["waveform"]
    return wav.unsqueeze(0) if wav.dim() == 2 else wav  # [C, T] -> [1, C, T]


def restore_audio(audio, info, fps):
    """Trim/zero-pad an AUDIO dict to exactly orig_frames / fps seconds."""
    wav, sr = _waveform(audio), int(audio["sample_rate"])
    target = int(round(info.orig_frames / float(fps) * sr))
    t = wav.shape[-1]
    if t > target:
        wav = wav[..., :target]
    elif t < target:
        wav = F.pad(wav, (0, target - t))
    return {"waveform": wav, "sample_rate": sr}


def prepare_audio(audio, info, fps):
    """Zero-pad (or trim) an AUDIO dict to padded_frames / fps seconds, matching the padded video."""
    wav, sr = _waveform(audio), int(audio["sample_rate"])
    target = int(round(info.padded_frames / float(fps) * sr))
    t = wav.shape[-1]
    if t > target:
        wav = wav[..., :target]
    elif t < target:
        wav = F.pad(wav, (0, target - t))
    return {"waveform": wav, "sample_rate": sr}


# ---------------------------------------------------------------- matte / temporal checks
def _small_gray(x, size=64, chunk=16):
    """[N,H,W,C] -> [N,size,size] float gray, chunked so 4K batches stay cheap."""
    outs = []
    for i in range(0, x.shape[0], chunk):
        c = x[i:i + chunk].float()
        g = c[..., :3].mean(-1) if c.shape[-1] >= 3 else c[..., 0]
        outs.append(F.adaptive_avg_pool2d(g.unsqueeze(1), (size, size)).squeeze(1))
    return torch.cat(outs)


def _lagged_corr(a, b, lag):
    """Pearson corr of a[t] vs b[t+lag] (None if undefined)."""
    if lag >= 0:
        x, y = a[: len(a) - lag], b[lag:]
    else:
        x, y = a[-lag:], b[: len(b) + lag]
    if len(x) < 4:
        return None
    x, y = x - x.mean(), y - y.mean()
    den = float(x.norm() * y.norm())
    return float((x * y).sum()) / den if den > 1e-12 else None


def temporal_check(source, result, max_lag=4, lag_margin=0.1, spike_thr=0.15):
    """Frame-to-frame sanity checks of a generated matte against its source.

    Returns (problems, warnings, info). `problems` are hard failures (wrong count, NaN/Inf, out of range,
    frames shifted in time, duplicated frames, isolated empty frames); `warnings` are soft (flicker spikes,
    static clips). Frame indices are 0-based.
    """
    problems, warnings, info = [], [], {}
    n = result.shape[0]
    if source.shape[0] != n:
        return [f"frame count {source.shape[0]} (source) vs {n} (result)"], warnings, info

    # NaN / Inf / range
    if not bool(torch.isfinite(result).all()):
        bad = [t for t in range(n) if not bool(torch.isfinite(result[t]).all())]
        problems.append(f"non-finite values (NaN/Inf) in frames {bad[:20]}")
    lo, hi = float(result.amin()), float(result.amax())
    info["range"] = (lo, hi)
    if lo < -0.01 or hi > 1.01:
        problems.append(f"values out of 0..1 range: min {lo:.3f} max {hi:.3f}")
    if n < 2:
        return problems, warnings, info

    # per-frame coverage (mean) and emptiness (max)
    means = torch.stack([result[t].float().mean() for t in range(n)])
    maxs = torch.stack([result[t].float().amax() for t in range(n)])
    info["mean_coverage"] = [round(float(m), 4) for m in means[: min(n, 5)]]

    # exact duplicate frames in the result that are NOT duplicates in the source
    dups = [t for t in range(1, n)
            if torch.equal(result[t], result[t - 1]) and not torch.equal(source[t], source[t - 1])]
    if dups:
        problems.append(f"duplicated result frames (held/repeated) at {dups[:20]}")

    # isolated dropouts: empty frame between frames that have matte content
    drops = []
    for t in range(n):
        if float(maxs[t]) < 1e-3:
            nb = [float(means[u]) for u in (t - 1, t + 1) if 0 <= u < n]
            if nb and all(v > 0.02 for v in nb):
                drops.append(t)
    if drops:
        problems.append(f"isolated empty (dropout) frames at {drops[:20]}")

    # flicker spikes (soft)
    spikes = []
    for t in range(1, n - 1):
        med = float(means[t - 1] + means[t + 1]) / 2
        if abs(float(means[t]) - med) > spike_thr and abs(float(means[t - 1] - means[t + 1])) < spike_thr / 2:
            spikes.append(t)
    if spikes:
        warnings.append(f"coverage spikes (possible flicker) at {spikes[:20]}")

    # temporal alignment: motion energy of result must line up with motion energy of source
    gs, gr = _small_gray(source), _small_gray(result)
    ms = (gs[1:] - gs[:-1]).abs().mean(dim=(1, 2))
    mr = (gr[1:] - gr[:-1]).abs().mean(dim=(1, 2))
    if n < 8 or float(ms.std()) < 1e-5:
        warnings.append("alignment check skipped (clip too short or source is static)")
    else:
        corrs = {l: _lagged_corr(ms, mr, l) for l in range(-max_lag, max_lag + 1)}
        valid = {l: c for l, c in corrs.items() if c is not None}
        info["lag_corr"] = {l: round(c, 3) for l, c in valid.items()}
        c0 = valid.get(0)
        if c0 is None:
            warnings.append("result has no motion; alignment could not be verified")
        else:
            best_lag = max(valid, key=valid.get)
            if best_lag != 0 and valid[best_lag] - c0 > lag_margin:
                problems.append(f"frames shifted in time: result best matches source at lag {best_lag:+d} "
                                f"(corr {valid[best_lag]:.2f} vs {c0:.2f} at lag 0)")
            info["best_lag"] = best_lag
            info["corr_at_0"] = round(c0, 3)
    return problems, warnings, info

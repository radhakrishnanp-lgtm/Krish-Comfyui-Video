"""Inpaint crop / stitch core (torch only, no ComfyUI imports, unit-testable).

Goal: edit a region of a big image with a model, then put the result back so that everything outside the edit is the
ORIGINAL pixels, bit for bit, and the seam is invisible.

Design (original implementation; the idea of crop -> model -> stitch is shared with other packs, e.g. lquesada's
Inpaint-CropAndStitch and HallettVisual's Smart Image Crop and Stitch, both studied while designing this):

  * exact geometry plan: the crop box has integer size and position, the model size is a multiple of N, and the two
    scale factors (sx, sy) are stored, so the stitch applies the exact inverse of what the crop did.
  * resampling picks its filter by DIRECTION (antialiased when shrinking, bicubic when enlarging); never 'area' for
    enlarging, which would be nearest-neighbour. Results are limited to the source range (no bicubic ringing, HDR safe).
  * the blend follows the mask shape (feather outward from the mask edge, exactly 0 beyond it), not the crop box.
  * optional sub-pixel alignment of the model result to the original (phase correlation on the untouched ring), because
    edit models often shift the picture by a fraction of a pixel.
  * optional low-frequency colour match estimated on the untouched ring and extended smoothly into the edit.
  * compositing uses orig + alpha * (gen - orig) and skips alpha == 0, so identical content stays bit-identical.

Images are [B,H,W,C] float tensors, masks [B,H,W] float tensors (white = edit). Nothing is clamped to 0..1.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

try:  # scipy ships with ComfyUI's requirements; fall back gracefully when it is absent
    from scipy import ndimage as _ndi
except Exception:  # pragma: no cover
    _ndi = None


# ------------------------------------------------------------------ small helpers
def to_bhwc(x):
    if x.ndim == 3:
        return x.unsqueeze(0)
    return x


def to_bhw(m):
    if m.ndim == 2:
        return m.unsqueeze(0)
    if m.ndim == 4 and m.shape[-1] == 1:
        return m.squeeze(-1)
    if m.ndim == 4 and m.shape[1] == 1:
        return m.squeeze(1)
    return m


def round_to_multiple(v, m):
    return max(m, int(round(float(v) / m)) * m)


def ceil_to_multiple(v, m):
    return max(m, int(math.ceil(float(v) / m - 1e-9)) * m)


# ------------------------------------------------------------------ mask preparation
def _propagate_rows(reach, free):
    """Spread `reach` along every horizontal run of free pixels in one vectorised pass."""
    h, w = free.shape
    run_id = torch.cumsum((~free).to(torch.int64), dim=1)
    key = (torch.arange(h, device=free.device)[:, None] * (w + 1) + run_id).flatten()
    mx = torch.zeros(h * (w + 1), device=free.device).scatter_reduce_(0, key, (reach & free).flatten().float(), "amax")
    return (mx[key] > 0).view(h, w) & free


def _fill_holes_one(m):
    """m: [H,W] bool numpy -> holes enclosed by the mask are filled (4-connected background, like scipy)."""
    if _ndi is not None:
        return _ndi.binary_fill_holes(m)
    # Fallback without scipy: flood the background in from the border by alternating row and column run propagation,
    # which needs only as many passes as the flood path has turns. Holes lie inside the mask's bounding box, so only that
    # box (plus a 1px border) is processed.
    ys, xs = np.nonzero(m)
    if ys.size == 0:
        return m
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sub = np.pad(m[y0:y1, x0:x1], 1, constant_values=False)
    free = torch.from_numpy(~sub)
    reach = torch.zeros_like(free)
    reach[0, 0] = True                                            # the padded border is background and connected
    while True:
        nxt = _propagate_rows(reach, free)
        nxt = _propagate_rows(nxt.T, free.T).T
        if torch.equal(nxt, reach):
            break
        reach = nxt
    out = m.copy()
    out[y0:y1, x0:x1] = ~reach[1:-1, 1:-1].numpy()
    return out


def _morph(mask_bhw, radius, grow):
    """Square dilate (grow) / erode, separable so big radii stay cheap."""
    if radius <= 0:
        return mask_bhw
    k = 2 * radius + 1
    x = mask_bhw.unsqueeze(1)
    if not grow:
        x = 1 - x
    x = F.max_pool2d(x, (1, k), 1, (0, radius))
    x = F.max_pool2d(x, (k, 1), 1, (radius, 0))
    if not grow:
        x = 1 - x
    return x.squeeze(1)


def clean_mask(mask_bhw, threshold=0.01, fill_holes=True, grow=0):
    """Binary edit region: values above `threshold` (so even very light greys count), holes filled, grown/shrunk."""
    core = (mask_bhw > float(threshold)).float()
    if fill_holes and core.any():
        filled = np.stack([_fill_holes_one(c.numpy() > 0.5) for c in core.detach().cpu()], axis=0)
        core = torch.from_numpy(filled.astype(np.float32)).to(core.device)
    if grow:
        core = _morph(core, abs(int(grow)), grow > 0)
    return core


def mask_bbox(core_bhw):
    """Union bounding box over the batch -> (x0, y0, x1, y1) exclusive, or None if the mask is empty."""
    any_hw = core_bhw.amax(dim=0) > 0.5
    ys = torch.nonzero(any_hw.any(dim=1)).flatten()
    xs = torch.nonzero(any_hw.any(dim=0)).flatten()
    if ys.numel() == 0 or xs.numel() == 0:
        return None
    return int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1


# ------------------------------------------------------------------ geometry plan
SIZE_MODES = ["megapixels", "longest_side", "fixed", "native (no resize)"]


def fit_megapixels(target_px, aspect, multiple):
    """(width, height), both multiples of `multiple`, whose area is as close to target_px as possible while keeping the
    aspect ratio (width / height) within a fraction of a percent."""
    th0 = round_to_multiple(math.sqrt(target_px / aspect), multiple)
    best, best_err = None, None
    for i in range(-6, 7):
        th = max(multiple, th0 + i * multiple)
        tw0 = round_to_multiple(target_px / th, multiple)
        for j in range(-3, 4):
            tw = max(multiple, tw0 + j * multiple)
            err = abs(tw * th - target_px) / target_px + abs(math.log((tw / th) / aspect))
            if best_err is None or err < best_err:
                best, best_err = (tw, th), err
    return best


def plan_crop(H, W, bbox, *, context_scale=1.3, context_pixels=32, size_mode="megapixels", megapixels=2.0,
              longest_side=1536, fixed_width=1024, fixed_height=1024, multiple=16, max_upscale=4.0,
              mp_unit="binary", shape="region"):
    """Integer crop box + model size + the exact scale factors. The box may extend past the image (padded later)."""
    x0, y0, x1, y1 = bbox
    bw, bh = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    rw = max(bw * float(context_scale), bw + 2.0 * context_pixels, 1.0)
    rh = max(bh * float(context_scale), bh + 2.0 * context_pixels, 1.0)

    if str(shape).startswith("square") and size_mode != "fixed":
        rw = rh = max(rw, rh)                                    # square crop: the region is centred in a square box
    if size_mode == "native (no resize)":
        tw, th = ceil_to_multiple(rw, multiple), ceil_to_multiple(rh, multiple)
        box_w, box_h = tw, th
    else:
        if size_mode == "fixed":
            tw, th = round_to_multiple(fixed_width, multiple), round_to_multiple(fixed_height, multiple)
            if rw / rh > tw / th:           # region too wide for the target shape: grow its height
                rh = rw * th / tw
            else:
                rw = rh * tw / th
            s = tw / rw
        else:
            if size_mode == "longest_side":
                s = float(longest_side) / max(rw, rh)
            else:
                target = float(megapixels) * (1048576.0 if str(mp_unit).startswith("binary") else 1e6)
                s = math.sqrt(target / (rw * rh))
            if size_mode == "megapixels" and s > float(max_upscale):
                grow = s / float(max_upscale)                    # tiny edit region: take in more surroundings instead of
                rw, rh = rw * grow, rh * grow                    # upscaling past max_upscale, so the requested size is still met
                s = float(max_upscale)
            s = min(s, float(max_upscale))
            if size_mode == "megapixels":
                tw, th = fit_megapixels(target, rw / rh, multiple)   # the multiple-of-N size nearest the requested pixel count
                s = math.sqrt(tw * th / (rw * rh))
            else:
                tw, th = round_to_multiple(rw * s, multiple), round_to_multiple(rh * s, multiple)
        box_w, box_h = max(1, round(tw / s)), max(1, round(th / s))
    box_w, box_h = max(box_w, bw), max(box_h, bh)               # the box always contains the whole edit region

    def place(c, size, limit):
        if size > limit:                                          # larger than the image: centre it, pad the rest
            return int(round(c - size / 2.0))
        return int(min(max(round(c - size / 2.0), 0), limit - size))

    bx, by = place(cx, box_w, W), place(cy, box_h, H)
    return {
        "H": H, "W": W, "bbox": (x0, y0, x1, y1),
        "x": bx, "y": by, "bw": box_w, "bh": box_h, "tw": tw, "th": th,
        "sx": tw / box_w, "sy": th / box_h,
        "pad": (max(0, -bx), max(0, -by), max(0, bx + box_w - W), max(0, by + box_h - H)),
    }


def extract_box(img_bhwc, plan, pad_mode="reflect"):
    """Crop the plan's box out of an image, padding where the box leaves the image."""
    x, y, bw, bh = plan["x"], plan["y"], plan["bw"], plan["bh"]
    H, W = img_bhwc.shape[1:3]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + bw), min(H, y + bh)
    crop = img_bhwc[:, y0:y1, x0:x1, :]
    pl, pt, pr, pb = plan["pad"]
    if pl or pt or pr or pb:
        t = crop.permute(0, 3, 1, 2)
        mode = pad_mode if pad_mode == "reflect" and max(pl, pr) < t.shape[3] and max(pt, pb) < t.shape[2] else "replicate"
        t = F.pad(t, (pl, pr, pt, pb), mode=mode)
        crop = t.permute(0, 2, 3, 1)
    return crop.contiguous()


def extract_box_mask(m_bhw, plan):
    """Same crop for masks; anything outside the image counts as 'not edited' (zero)."""
    x, y, bw, bh = plan["x"], plan["y"], plan["bw"], plan["bh"]
    H, W = m_bhw.shape[1:3]
    out = torch.zeros((m_bhw.shape[0], bh, bw), dtype=m_bhw.dtype, device=m_bhw.device)
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + bw), min(H, y + bh)
    out[:, y0 - y:y1 - y, x0 - x:x1 - x] = m_bhw[:, y0:y1, x0:x1]
    return out


# ------------------------------------------------------------------ resampling
RESAMPLE = ["bicubic", "bilinear", "area"]


def resize_img(img_bhwc, th, tw, mode="bicubic"):
    """Direction-aware resize. Shrinking is antialiased; enlarging is bicubic/bilinear (never 'area' = nearest)."""
    h, w = img_bhwc.shape[1:3]
    if (h, w) == (th, tw):
        return img_bhwc
    t = img_bhwc.permute(0, 3, 1, 2).float()
    lo, hi = t.amin(dim=(0, 2, 3), keepdim=True), t.amax(dim=(0, 2, 3), keepdim=True)
    shrinking = th <= h and tw <= w
    if mode == "area" and shrinking:
        t = F.interpolate(t, size=(th, tw), mode="area")
    else:
        m = "bilinear" if mode == "bilinear" else "bicubic"
        t = F.interpolate(t, size=(th, tw), mode=m, align_corners=False, antialias=(th < h or tw < w))
    return torch.minimum(torch.maximum(t, lo), hi).permute(0, 2, 3, 1).contiguous()


def resize_mask(m_bhw, th, tw):
    if m_bhw.shape[1:3] == (th, tw):
        return m_bhw
    t = m_bhw.unsqueeze(1).float()
    t = F.interpolate(t, size=(th, tw), mode="bilinear", align_corners=False, antialias=(th < t.shape[2] or tw < t.shape[3]))
    return t.squeeze(1).clamp(0, 1)


# ------------------------------------------------------------------ blur / feather
def gaussian_blur(x_bhw, sigma):
    """Separable gaussian on [B,H,W]; replicate padding so edges do not darken."""
    if sigma <= 0.05:
        return x_bhw
    r = max(1, int(math.ceil(3 * sigma)))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, dtype=torch.float32, device=x_bhw.device) / sigma) ** 2)
    k = (k / k.sum())
    x = x_bhw.unsqueeze(1).float()
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))
    return x.squeeze(1)


def auto_feather(bbox):
    x0, y0, x1, y1 = bbox
    return int(min(64, max(6, round(min(x1 - x0, y1 - y0) * 0.10))))


def _outside_distance(core_bhw):
    """Euclidean distance (px) from every pixel to the nearest mask pixel (0 inside); None without scipy."""
    if _ndi is None:
        return None
    d = np.stack([_ndi.distance_transform_edt(c < 0.5) for c in core_bhw.detach().cpu().numpy()], axis=0)
    return torch.from_numpy(d.astype(np.float32)).to(core_bhw.device)


def build_alpha(core_bhw, soft_bhw, mode="mask_feather", feather=24):
    """Blend weights in the crop box. 1 inside the edit, falls smoothly to EXACTLY 0 within `feather` pixels outside it."""
    core = core_bhw.float()
    if mode == "hard" or feather <= 0:
        a = core
    elif mode == "box_feather":
        a = torch.ones_like(core)
        m = int(feather)
        a[:, :m, :], a[:, -m:, :], a[:, :, :m], a[:, :, -m:] = 0, 0, 0, 0
        a = gaussian_blur(a, feather / 2.5)
    else:
        d = _outside_distance(core)
        if d is not None:
            t = (1.0 - d / float(feather)).clamp(0, 1)            # 1 on the mask edge, 0 at `feather` px away
            a = t * t * (3.0 - 2.0 * t)                           # smoothstep: zero slope at both ends, so no visible kink
        else:                                                     # no scipy: blur-based falloff with the same reach
            floor = 0.012                                         # value of 2*Q(2.5): removes the Gaussian tail
            a = (2.0 * gaussian_blur(core, feather / 2.5)).clamp(0, 1)
            a = ((a - floor) / (1 - floor)).clamp(0, 1)
            a = a * _morph(core, int(math.ceil(feather)), True)   # strictly zero beyond the feather reach
        a = torch.maximum(a, core)
    if soft_bhw is not None:
        a = torch.maximum(a, soft_bhw.float())
    return a


# ------------------------------------------------------------------ alignment (phase correlation)
def _luma(img_bhwc):
    if img_bhwc.shape[-1] >= 3:
        return (0.299 * img_bhwc[..., 0] + 0.587 * img_bhwc[..., 1] + 0.114 * img_bhwc[..., 2]).float()
    return img_bhwc[..., 0].float()


def _lk_refine(a, b, w, dx, dy, iters=5):
    """Gradient (Lucas-Kanade) refinement of a translation on single-channel [1,h,w] tensors.

    Finds (dx, dy) minimising sum w * (a(x) - b(x + d))^2, starting from (dx, dy). Accurate to a few hundredths of a pixel
    on textured content; returns the refined shift."""
    for _ in range(iters):
        warped = shift_image(b[..., None], dx, dy)[..., 0]
        gx = torch.zeros_like(warped)
        gy = torch.zeros_like(warped)
        gx[:, :, 1:-1] = (warped[:, :, 2:] - warped[:, :, :-2]) * 0.5
        gy[:, 1:-1, :] = (warped[:, 2:, :] - warped[:, :-2, :]) * 0.5
        err = a - warped
        wm = w.clone()
        wm[:, :2, :], wm[:, -2:, :], wm[:, :, :2], wm[:, :, -2:] = 0, 0, 0, 0
        axx, axy, ayy = (wm * gx * gx).sum(), (wm * gx * gy).sum(), (wm * gy * gy).sum()
        bx, by = (wm * gx * err).sum(), (wm * gy * err).sum()
        det = axx * ayy - axy * axy
        if float(det) < 1e-12:
            break
        ddx, ddy = (ayy * bx - axy * by) / det, (axx * by - axy * bx) / det
        dx, dy = dx + float(ddx), dy + float(ddy)
        if abs(float(ddx)) < 1e-3 and abs(float(ddy)) < 1e-3:
            break
    return dx, dy


def _normalize_ring(x, w):
    """High-pass x on the weighted ring and scale it to unit weighted std, so a gain/offset change between the original
    and the model result does not bias the translation estimate."""
    sigma = max(3.0, min(x.shape[1:]) / 40.0)
    den = gaussian_blur(w, sigma).clamp_min(1e-6)
    hp = (x - gaussian_blur(x * w, sigma) / den) * (den > 1e-3)
    std = torch.sqrt((hp * hp * w).sum() / w.sum().clamp_min(1e-6)).clamp_min(1e-8)
    return hp / std


def estimate_shift(orig_box, gen_box, weight_bhw, max_dim=1536):
    """Translation (dx, dy) in pixels to APPLY to gen (sample it at x + dx, y + dy) so that it lines up with orig.

    Phase correlation finds the integer shift (robust), a gradient refinement makes it sub-pixel accurate. Only the
    untouched ring (weight > 0) is used. Returns (dx, dy, confidence, improvement): confidence is the phase-correlation
    peak, improvement is the fraction by which the ring mismatch shrinks when the shift is applied (negative = worse).
    """
    a, b, w = _luma(orig_box)[:1], _luma(gen_box)[:1], weight_bhw[:1].float()
    h, wd = a.shape[1:]
    k = max(1, int(math.ceil(max(h, wd) / max_dim)))
    if k > 1:
        a, b, w = (F.avg_pool2d(t.unsqueeze(1), k, k, ceil_mode=True).squeeze(1) for t in (a, b, w))
    h, wd = a.shape[1:]
    if h < 16 or wd < 16 or float(w.sum()) < 64:
        return 0.0, 0.0, 0.0, 0.0
    a, b = _normalize_ring(a, w), _normalize_ring(b, w)       # removes gain / offset / slow shading differences
    ws = w / w.sum()
    ma, mb = (a * ws).sum(), (b * ws).sum()
    hann = (torch.hann_window(h, periodic=False, device=a.device)[:, None] * torch.hann_window(wd, periodic=False, device=a.device)[None, :])[None]
    win = w * hann
    fa, fb = torch.fft.rfft2((a - ma) * win), torch.fft.rfft2((b - mb) * win)
    cross = fa * torch.conj(fb)
    cross = cross / (cross.abs() + 1e-8)
    r = torch.fft.irfft2(cross, s=(h, wd))[0]
    idx = int(torch.argmax(r))
    py, px = divmod(idx, wd)
    peak = float(r[py, px])
    sy = py - (h if py > h / 2 else 0)
    sx = px - (wd if px > wd / 2 else 0)
    dx, dy = _lk_refine(a, b, w, float(-sx), float(-sy))
    base = float((w * (a - b).abs()).sum() / w.sum())
    after = float((w * (a - shift_image(b[..., None], dx, dy)[..., 0]).abs()).sum() / w.sum())
    improvement = 0.0 if base <= 1e-12 else (base - after) / base
    return float(dx * k), float(dy * k), peak, improvement


def shift_image(img_bhwc, dx, dy):
    """Return img sampled at (x + dx, y + dy) with bicubic interpolation (edges replicate)."""
    B, H, W, C = img_bhwc.shape
    t = img_bhwc.permute(0, 3, 1, 2).float()
    ys, xs = torch.meshgrid(torch.arange(H, device=t.device, dtype=torch.float32),
                            torch.arange(W, device=t.device, dtype=torch.float32), indexing="ij")
    gx = (xs + dx + 0.5) / W * 2 - 1
    gy = (ys + dy + 0.5) / H * 2 - 1
    grid = torch.stack([gx, gy], dim=-1)[None].expand(B, -1, -1, -1)
    out = F.grid_sample(t, grid, mode="bicubic", padding_mode="border", align_corners=False)
    lo, hi = t.amin(dim=(0, 2, 3), keepdim=True), t.amax(dim=(0, 2, 3), keepdim=True)
    return torch.minimum(torch.maximum(out, lo), hi).permute(0, 2, 3, 1).contiguous()


def warp_affine(img_bhwc, tx, ty, m=(0.0, 0.0, 0.0, 0.0)):
    """Sample img at x' = x + tx + m0*u + m1*v, y' = y + ty + m2*u + m3*v, where (u, v) are pixel offsets from the centre
    divided by half the image size (so m is a dimensionless scale / rotation term). Bicubic, edges replicate."""
    B, H, W, C = img_bhwc.shape
    t = img_bhwc.permute(0, 3, 1, 2).float()
    ys, xs = torch.meshgrid(torch.arange(H, device=t.device, dtype=torch.float32),
                            torch.arange(W, device=t.device, dtype=torch.float32), indexing="ij")
    u, v = (xs - (W - 1) / 2) / (W / 2), (ys - (H - 1) / 2) / (H / 2)
    sx = xs + tx + m[0] * u + m[1] * v
    sy = ys + ty + m[2] * u + m[3] * v
    grid = torch.stack([(sx + 0.5) / W * 2 - 1, (sy + 0.5) / H * 2 - 1], dim=-1)[None].expand(B, -1, -1, -1)
    out = F.grid_sample(t, grid, mode="bicubic", padding_mode="border", align_corners=False)
    lo, hi = t.amin(dim=(0, 2, 3), keepdim=True), t.amax(dim=(0, 2, 3), keepdim=True)
    return torch.minimum(torch.maximum(out, lo), hi).permute(0, 2, 3, 1).contiguous()


def _lk_affine(a, b, w, p, iters=8):
    """Gauss-Newton refinement of p = (tx, ty, m0, m1, m2, m3) minimising sum w * (a - warp(b, p))^2 on [1,h,w] tensors."""
    h, wd = a.shape[1:]
    ys, xs = torch.meshgrid(torch.arange(h, device=a.device, dtype=torch.float32),
                            torch.arange(wd, device=a.device, dtype=torch.float32), indexing="ij")
    u, v = ((xs - (wd - 1) / 2) / (wd / 2))[None], ((ys - (h - 1) / 2) / (h / 2))[None]
    wm = w.clone()
    wm[:, :2, :], wm[:, -2:, :], wm[:, :, :2], wm[:, :, -2:] = 0, 0, 0, 0
    best_err, best_p = None, list(p)
    for _ in range(iters):
        warped = warp_affine(b[..., None], *p[:2], p[2:])[..., 0]
        err_map = a - warped
        e = float((wm * err_map * err_map).sum())
        if best_err is None or e < best_err:
            best_err, best_p = e, list(p)
        elif e > best_err:
            break
        gx, gy = torch.zeros_like(warped), torch.zeros_like(warped)
        gx[:, :, 1:-1] = (warped[:, :, 2:] - warped[:, :, :-2]) * 0.5
        gy[:, 1:-1, :] = (warped[:, 2:, :] - warped[:, :-2, :]) * 0.5
        J = torch.stack([gx, gy, gx * u, gx * v, gy * u, gy * v], dim=0).reshape(6, -1)
        wf, ef = wm.reshape(-1), err_map.reshape(-1)
        H = (J * wf) @ J.T + 1e-6 * torch.eye(6, device=a.device)
        g = (J * wf) @ ef
        try:
            dp = torch.linalg.solve(H.double(), g.double()).float()
        except Exception:
            break
        p = [p[i] + float(dp[i]) for i in range(6)]
        if float(dp[:2].abs().max()) < 1e-3 and float(dp[2:].abs().max()) < 1e-4:
            break
    warped = warp_affine(b[..., None], *p[:2], p[2:])[..., 0]
    e = float((wm * (a - warped) ** 2).sum())
    if best_err is not None and e > best_err:
        p = best_p
    return p


def estimate_warp(orig_box, gen_box, weight_bhw, max_dim=1536):
    """Shift + scale + rotation (affine) to APPLY to gen (via warp_affine) so it lines up with orig on the untouched ring.

    Starts from the phase-correlation shift, refines translation, then fits the full affine coarse-to-fine.
    Returns (tx, ty, m, confidence, improvement, max_corner_px)."""
    a, b, w = _luma(orig_box)[:1], _luma(gen_box)[:1], weight_bhw[:1].float()
    h0, w0 = a.shape[1:]
    k = max(1, int(math.ceil(max(h0, w0) / max_dim)))
    if k > 1:
        a, b, w = (F.avg_pool2d(t.unsqueeze(1), k, k, ceil_mode=True).squeeze(1) for t in (a, b, w))
    h, wd = a.shape[1:]
    if h < 16 or wd < 16 or float(w.sum()) < 64:
        return 0.0, 0.0, (0.0, 0.0, 0.0, 0.0), 0.0, 0.0, 0.0
    a, b = _normalize_ring(a, w), _normalize_ring(b, w)
    ws = w / w.sum()
    ma, mb = (a * ws).sum(), (b * ws).sum()
    hann = (torch.hann_window(h, periodic=False, device=a.device)[:, None] * torch.hann_window(wd, periodic=False, device=a.device)[None, :])[None]
    fa, fb = torch.fft.rfft2((a - ma) * w * hann), torch.fft.rfft2((b - mb) * w * hann)
    cross = fa * torch.conj(fb)
    r = torch.fft.irfft2(cross / (cross.abs() + 1e-8), s=(h, wd))[0]
    idx = int(torch.argmax(r))
    py, px = divmod(idx, wd)
    peak = float(r[py, px])
    sy = py - (h if py > h / 2 else 0)
    sx = px - (wd if px > wd / 2 else 0)
    tx, ty = _lk_refine(a, b, w, float(-sx), float(-sy))
    p = [tx, ty, 0.0, 0.0, 0.0, 0.0]
    base = float((w * (a - b).abs()).sum() / w.sum())
    if min(h, wd) >= 96:                                         # coarse level first: wider capture range for zoom / rotation
        a2, b2, w2 = (F.avg_pool2d(t.unsqueeze(1), 2, 2, ceil_mode=True).squeeze(1) for t in (a, b, w))
        q = _lk_affine(a2, b2, w2, [p[0] / 2, p[1] / 2, 0, 0, 0, 0])
        p = [q[0] * 2, q[1] * 2, q[2] * 2, q[3] * 2, q[4] * 2, q[5] * 2]
    p = _lk_affine(a, b, w, p)
    after = float((w * (a - warp_affine(b[..., None], p[0], p[1], p[2:])[..., 0]).abs()).sum() / w.sum())
    improvement = 0.0 if base <= 1e-12 else (base - after) / base
    corners = [(sgn_u, sgn_v) for sgn_u in (-1, 1) for sgn_v in (-1, 1)]
    mx = max(math.hypot(p[0] + p[2] * cu + p[3] * cv, p[1] + p[4] * cu + p[5] * cv) for cu, cv in corners)
    # translate to full resolution: shifts scale by k; m is dimensionless so it carries over unchanged
    return p[0] * k, p[1] * k, (p[2] * k, p[3] * k, p[4] * k, p[5] * k), peak, improvement, mx * k


# ------------------------------------------------------------------ colour (low-frequency field from the untouched ring)
def lowfreq_correction(orig_box, gen_box, weight_bhw, sigma_px):
    """Smooth per-channel offset that makes gen match orig on the ring, extended smoothly into the edit."""
    B, H, W, C = gen_box.shape
    k = max(1, int(min(H, W) // 128))
    d = ((orig_box - gen_box).float() * weight_bhw.unsqueeze(-1)).permute(0, 3, 1, 2)
    w = weight_bhw.float().unsqueeze(1)
    if k > 1:
        d, w = F.avg_pool2d(d, k, k, ceil_mode=True), F.avg_pool2d(w, k, k, ceil_mode=True)
    s = max(1.0, sigma_px / k)
    num = gaussian_blur(d.reshape(-1, *d.shape[2:]), s).reshape(d.shape)
    den = gaussian_blur(w[:, 0], s).unsqueeze(1)
    field = num / den.clamp_min(1e-6)
    field = field * (den / 0.05).clamp(0, 1)                    # no nearby ring -> no correction
    if k > 1:
        field = F.interpolate(field, size=(H, W), mode="bicubic", align_corners=False)
    return field.permute(0, 2, 3, 1)


# ------------------------------------------------------------------ letterbox protection
def detect_bars(img_bhwc, flat_tol=0.004, dark_max=0.05, max_frac=0.4):
    """Flat dark letterbox / pillarbox bars of the first image: (top, bottom, left, right, level) in pixels.

    A bar is a run of rows / columns, counted from the border, whose pixels are all (nearly) the same dark value."""
    x = img_bhwc[0].float()
    H, W, _ = x.shape

    def run(lines_max, lines_min, lines_mean, limit):
        n = 0
        for i in range(limit):
            if float(lines_max[i] - lines_min[i]) < flat_tol and float(lines_mean[i]) < dark_max:
                n += 1
            else:
                break
        return 0 if n >= limit else n          # a run that fills the cap is a flat image, not a bar

    rmax, rmin, rmean = x.amax(dim=(1, 2)), x.amin(dim=(1, 2)), x.mean(dim=(1, 2))
    lim_h = int(H * max_frac)
    top = run(rmax, rmin, rmean, lim_h)
    bottom = run(rmax.flip(0), rmin.flip(0), rmean.flip(0), lim_h)
    pic = x[top:H - bottom] if top + bottom < H else x
    cmax, cmin, cmean = pic.amax(dim=(0, 2)), pic.amin(dim=(0, 2)), pic.mean(dim=(0, 2))
    lim_w = int(W * max_frac)
    left = run(cmax, cmin, cmean, lim_w)
    right = run(cmax.flip(0), cmin.flip(0), cmean.flip(0), lim_w)
    level = 0.0
    if top:
        level = float(rmean[:top].mean())
    elif bottom:
        level = float(rmean[H - bottom:].mean())
    elif left:
        level = float(cmean[:left].mean())
    elif right:
        level = float(cmean[W - right:].mean())
    return top, bottom, left, right, level


def bar_guard(o, g, bars, x0, y0, H, W, margin=4, ramp=6, band_frac=0.02):
    """Multiplier [1,h,w] for the blend alpha so letterbox bars survive exactly.

    o / g are the original and the model result for the image region starting at (x0, y0). Inside a bar the multiplier is 0
    (original kept bit-exact). The model often draws a bar edge a few pixels off, bleeding black into the picture (or
    picture into the bar): wherever the result is bar-black but the original is not, the blend is held back from the
    bar edge out to that distance plus `margin`, then eased in over `ramp` pixels."""
    top, bottom, left, right, level = bars
    h, w = o.shape[1:3]
    lo, lg = _luma(o), _luma(g)
    ys = torch.arange(h, device=o.device, dtype=torch.float32)[:, None] + y0
    xs = torch.arange(w, device=o.device, dtype=torch.float32)[None, :] + x0
    band = max(16, int(band_frac * min(H, W)))
    dark = (lg < level + 0.015) & (lo > level + 0.03)
    mult = torch.ones((1, h, w), device=o.device)
    sides = [(top, (ys - top).expand(h, w)), (bottom, ((H - 1 - bottom) - ys).expand(h, w)),
             (left, (xs - left).expand(h, w)), (right, ((W - 1 - right) - xs).expand(h, w))]
    for extent, dist in sides:
        if extent <= 0:
            continue
        d = dist[None]
        hit = dark & (d >= 0) & (d < band)
        reach = int(d[hit].max()) + 1 if bool(hit.any()) else 0
        if reach > 0:
            t = ((d - (reach + margin)) / float(ramp)).clamp(0, 1)
            m = torch.where(d < 0, torch.zeros_like(t), t * t * (3 - 2 * t))
        else:
            m = (d >= 0).float()
        mult = mult * m
    return mult


# ------------------------------------------------------------------ crop + stitch
def crop(image, mask, *, threshold=0.01, fill_holes=True, grow=0, edge_pad="reflect", resample="bicubic", **plan_kw):
    """Returns (stitcher dict, cropped_image, cropped_mask, info str). stitcher is None-like (bypass) for an empty mask."""
    img = to_bhwc(image).contiguous()
    B, H, W, C = img.shape
    note = ""
    m = to_bhw(mask).float()
    if m.shape[1:] != (H, W):
        note = f"mask resized from {m.shape[2]}x{m.shape[1]} to {W}x{H}. "
        if abs((m.shape[2] / m.shape[1]) / (W / H) - 1.0) > 0.01:
            note = (f"WARNING: the mask is {m.shape[2]}x{m.shape[1]} but the image is {W}x{H} - a different aspect ratio, so the mask is "
                    f"STRETCHED and will not line up with the picture. Export the mask at the image size. ")
        m = resize_mask(m, H, W)
    if m.shape[0] not in (1, B):
        m = m[:1]
    m = m.expand(B, -1, -1).contiguous() if m.shape[0] == 1 else m
    core = clean_mask(m, threshold, fill_holes, grow)
    bbox = mask_bbox(core)
    if bbox is None:
        st = {"bypass": True, "original": img, "H": H, "W": W}
        return st, img, torch.zeros((B, H, W), dtype=torch.float32, device=img.device), note + "empty mask: nothing to crop, the stitch passes the image through"
    plan = plan_crop(H, W, bbox, **plan_kw)
    box = extract_box(img, plan, edge_pad)
    cropped = resize_img(box, plan["th"], plan["tw"], resample)
    cmask = resize_mask(extract_box_mask(core, plan), plan["th"], plan["tw"])
    bars = [detect_bars(img[i:i + 1]) for i in range(B)]
    st = {"bypass": False, "original": img, "core": core, "soft": m.clamp(0, 1), "plan": plan, "edge_pad": edge_pad,
          "H": H, "W": W, "threshold": threshold, "bars": bars}
    info = (f"{note}mask box {bbox[2]-bbox[0]}x{bbox[3]-bbox[1]} at ({bbox[0]},{bbox[1]}); crop box {plan['bw']}x{plan['bh']} at "
            f"({plan['x']},{plan['y']}) -> model size {plan['tw']}x{plan['th']} = {plan['tw'] * plan['th'] / 1048576:.2f} MP (scale {plan['sx']:.4f} x {plan['sy']:.4f}); "
            f"padding l/t/r/b {plan['pad']}")
    if any(sum(bb[:4]) for bb in bars):
        t_, b_, l_, r_, _ = bars[0]
        info += f"; letterbox bars detected (top {t_}, bottom {b_}, left {l_}, right {r_} px): kept exactly"
    return st, cropped, cmask, info


def stitch(stitcher, generated, *, blend_mode="mask_feather", feather_mode="auto", feather_pixels=24, align="affine",
           max_shift_px=64, min_shift_px=0.05, min_confidence=0.08, color_match="low_frequency", color_amount=1.0,
           color_sigma=0, ring_margin=0, resample="bicubic", protect_letterbox=True):
    """Returns (image, blend_mask [B,H,W], report str)."""
    orig = stitcher["original"]
    B, H, W, C = orig.shape
    if stitcher.get("bypass"):
        return orig, torch.zeros((B, H, W), dtype=torch.float32, device=orig.device), "bypass: empty mask, image unchanged"
    plan, core, soft = stitcher["plan"], stitcher["core"], stitcher["soft"]
    gen_all = to_bhwc(generated)[..., :C].float()                    # edit models may decode RGBA: keep the original's channels
    out = orig.clone()
    alpha_full = torch.zeros((B, H, W), dtype=torch.float32, device=orig.device)
    feather = auto_feather(plan["bbox"]) if feather_mode == "auto" else int(feather_pixels)
    feather = min(feather, max(plan["bw"], plan["bh"]) // 2)
    lines = [f"feather {feather}px"]
    bw, bh, tw, th = plan["bw"], plan["bh"], plan["tw"], plan["th"]
    x0, y0, x1, y1 = max(0, plan["x"]), max(0, plan["y"]), min(W, plan["x"] + bw), min(H, plan["y"] + bh)
    ox, oy = x0 - plan["x"], y0 - plan["y"]

    for b in range(B):
        gen = gen_all[b % gen_all.shape[0]][None]
        if gen.shape[1:3] != (th, tw):
            lines.append(f"[{b}] model result {gen.shape[2]}x{gen.shape[1]} resized to the crop size {tw}x{th}")
            gen = resize_img(gen, th, tw, resample)
        gen_box = resize_img(gen, bh, bw, resample)                  # exact inverse of the crop's resize
        orig_box = extract_box(orig[b:b + 1], plan, stitcher["edge_pad"])
        core_box = extract_box_mask(core[b:b + 1], plan)
        soft_box = extract_box_mask(soft[b:b + 1], plan)
        margin = int(ring_margin) if ring_margin > 0 else max(4, feather)
        ring = 1.0 - _morph(core_box, margin, True)

        if align == "affine":
            tx, ty, mm, conf, gain, mx = estimate_warp(orig_box, gen_box, ring)
            mag = max(math.hypot(tx, ty), mx)
            if (conf >= min_confidence or gain >= 0.15) and gain >= 0.05 and min_shift_px <= mag <= max_shift_px:
                gen_box = warp_affine(gen_box, tx, ty, mm)
                zoom = 1.0 + 0.5 * (mm[0] / (bw / 2) + mm[3] / (bh / 2))
                rot = math.degrees(0.5 * (mm[2] / (bw / 2) - mm[1] / (bh / 2)))
                lines.append(f"[{b}] aligned the result: shift ({tx:+.2f}, {ty:+.2f}) px, zoom {zoom:.4f}, rotation {rot:+.3f} deg, "
                             f"max {mx:.1f} px at the box corners (confidence {conf:.2f}, ring mismatch -{gain:.0%})")
            else:
                lines.append(f"[{b}] no alignment applied (measured {tx:+.2f}, {ty:+.2f} px, max {mx:.1f} px, confidence {conf:.2f}, gain {gain:+.0%})")
        elif align == "translation":
            dx, dy, conf, gain = estimate_shift(orig_box, gen_box, ring)
            mag = math.hypot(dx, dy)
            if conf >= min_confidence and gain >= 0.05 and min_shift_px <= mag <= max_shift_px:
                gen_box = shift_image(gen_box, dx, dy)
                lines.append(f"[{b}] aligned the result by ({dx:+.2f}, {dy:+.2f}) px (confidence {conf:.2f}, ring mismatch -{gain:.0%})")
            else:
                lines.append(f"[{b}] no alignment applied (measured {dx:+.2f}, {dy:+.2f} px, confidence {conf:.2f}, gain {gain:+.0%})")
        if color_match == "low_frequency" and color_amount > 0:
            sigma = color_sigma if color_sigma > 0 else max(24.0, 0.1 * min(bw, bh))
            field = lowfreq_correction(orig_box, gen_box, ring, sigma)
            gen_box = gen_box + field * float(color_amount)
            lines.append(f"[{b}] colour field applied (max |offset| {float(field.abs().max()):.4f})")

        alpha = build_alpha(core_box, soft_box, blend_mode, feather)
        a = alpha[:, oy:oy + (y1 - y0), ox:ox + (x1 - x0)]
        g = gen_box[:, oy:oy + (y1 - y0), ox:ox + (x1 - x0), :]
        o = out[b:b + 1, y0:y1, x0:x1, :]
        bars = (stitcher.get("bars") or [None] * B)[b]
        if protect_letterbox and bars is not None and sum(bars[:4]) > 0:
            gm = bar_guard(o, g, bars, x0, y0, H, W, margin=max(4, feather // 4))
            if float(gm.min()) < 1.0:
                a = a * gm
                lines.append(f"[{b}] letterbox kept exact; blend held back from the bar edge by up to "
                             f"{int((gm[0] < 0.5).float().sum(dim=0).max())} px")
        blended = o + a.unsqueeze(-1) * (g - o)                      # lerp form: identical content stays bit-identical
        out[b:b + 1, y0:y1, x0:x1, :] = torch.where(a.unsqueeze(-1) > 0, blended, o)
        alpha_full[b, y0:y1, x0:x1] = a[0]
    return out, alpha_full, "; ".join(lines)


# ------------------------------------------------------------------ preview
def preview_overlay(image_bhwc, stitcher, max_side=1024):
    """Small picture of the first image: edit region tinted red, crop box outlined in green."""
    img = image_bhwc[:1].float()
    H, W = img.shape[1:3]
    s = min(1.0, max_side / max(H, W))
    ph, pw = max(1, round(H * s)), max(1, round(W * s))
    small = resize_img(img, ph, pw, "bicubic" if s >= 1 else "area").clamp(0, 1).clone()
    if stitcher.get("bypass"):
        return small
    core = resize_mask(stitcher["core"][:1], ph, pw)
    red = torch.tensor([1.0, 0.0, 0.0], device=small.device)[: small.shape[-1]]
    small = small + core.unsqueeze(-1) * 0.35 * (red - small)
    p = stitcher["plan"]
    x0, y0 = max(0, round(p["x"] * s)), max(0, round(p["y"] * s))
    x1, y1 = min(pw, round((p["x"] + p["bw"]) * s)), min(ph, round((p["y"] + p["bh"]) * s))
    green = torch.tensor([0.0, 1.0, 0.0], device=small.device)[: small.shape[-1]]
    if x1 - x0 > 4 and y1 - y0 > 4:
        small[:, y0:y0 + 2, x0:x1], small[:, y1 - 2:y1, x0:x1] = green, green
        small[:, y0:y1, x0:x0 + 2], small[:, y0:y1, x1 - 2:x1] = green, green
    return small

import importlib.util
import itertools
import math
import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("inpaint_core", ROOT / "krish_frame_conform" / "inpaint_core.py")
ic = importlib.util.module_from_spec(spec)
sys.modules["inpaint_core"] = ic
spec.loader.exec_module(ic)


def make_image(H, W, B=1, seed=0, hdr=40.0):
    """Smooth, textured, HDR-range image (values up to ~hdr, a few negatives)."""
    g = torch.Generator().manual_seed(seed)
    n = torch.rand(B, 3, H + 8, W + 8, generator=g)
    n = torch.nn.functional.avg_pool2d(n, 5, 1, 2)[:, :, 4:-4, 4:-4]
    yy, xx = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing="ij")
    ramp = (0.3 + 0.7 * xx * yy)[None, None]
    return ((n * 0.6 + ramp) * (hdr / 2) - 0.05).permute(0, 2, 3, 1).contiguous()


def make_smooth(H, W, hdr=8.0):
    """Low-frequency content only, so resampling quality (not lost detail) is what a test measures."""
    yy, xx = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing="ij")
    v = 1.5 + torch.sin(6.0 * xx + 1.0) + 0.8 * torch.cos(5.0 * yy - 0.5) + 0.6 * torch.sin(4.0 * (xx + yy))
    return (torch.stack([v, v * 0.8 + 0.3, v * 0.5 + 0.9], dim=-1) * (hdr / 4.0))[None].contiguous()


def disk_mask(H, W, cy, cx, r, B=1):
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    return (((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r).float()[None].repeat(B, 1, 1)


def distance_from_mask(core_hw):
    from scipy import ndimage
    return torch.from_numpy(ndimage.distance_transform_edt(1 - core_hw.numpy())).float()


# ------------------------------------------------------------------ mask preparation
def test_threshold_detects_light_greys_and_fills_holes_and_grows():
    m = torch.zeros(1, 40, 40)
    m[:, 10:30, 10:30] = 0.02                                      # very light grey
    m[:, 18:22, 18:22] = 0.0                                       # a hole
    core = ic.clean_mask(m, threshold=0.01, fill_holes=True, grow=0)
    assert float(core.sum()) == 20 * 20 and bool((core[:, 18:22, 18:22] == 1).all())
    assert float(ic.clean_mask(m, 0.05, True, 0).sum()) == 0                 # above threshold: nothing
    assert float(ic.clean_mask(m, 0.01, False, 0)[:, 20, 20]) == 0           # holes kept when fill_holes is off
    grown, shrunk = ic.clean_mask(m, 0.01, True, 3), ic.clean_mask(m, 0.01, True, -3)
    assert float(grown.sum()) == 26 * 26 and float(shrunk.sum()) == 14 * 14
    assert ic.mask_bbox(core) == (10, 10, 30, 30) and ic.mask_bbox(torch.zeros(1, 9, 9)) is None


# ------------------------------------------------------------------ geometry: any size, any mask position
SIZES = [(9, 7), (64, 64), (270, 480), (1080, 1920), (2304, 4096), (4000, 90), (90, 4000), (4321, 7777)]
MODES = ["megapixels", "longest_side", "fixed", "native (no resize)"]


@pytest.mark.parametrize("H,W", SIZES)
@pytest.mark.parametrize("size_mode", MODES)
def test_plan_invariants_for_any_size_and_mask_position(H, W, size_mode):
    for fx, fy, fw, fh in [(0.5, 0.5, 0.1, 0.1), (0.0, 0.0, 0.05, 0.05), (0.97, 0.97, 0.03, 0.03), (0.0, 0.0, 1.0, 1.0),
                           (0.3, 0.6, 0.5, 0.02), (0.45, 0.0, 0.02, 1.0)]:
        x0, y0 = int(fx * (W - 1)), int(fy * (H - 1))
        bbox = (x0, y0, min(W, x0 + max(1, int(fw * W))), min(H, y0 + max(1, int(fh * H))))
        p = ic.plan_crop(H, W, bbox, size_mode=size_mode, megapixels=2.0, longest_side=1536, fixed_width=1024,
                         fixed_height=640, multiple=16, max_upscale=4.0, context_scale=1.3, context_pixels=32)
        assert p["tw"] % 16 == 0 and p["th"] % 16 == 0 and p["tw"] >= 16 and p["th"] >= 16
        assert p["bw"] >= bbox[2] - bbox[0] and p["bh"] >= bbox[3] - bbox[1]                      # box holds the whole edit
        x0b, y0b = p["x"], p["y"]
        assert p["x"] <= bbox[0] and p["x"] + p["bw"] >= bbox[2] and p["y"] <= bbox[1] and p["y"] + p["bh"] >= bbox[3]
        assert abs(p["sx"] - p["tw"] / p["bw"]) < 1e-12 and abs(p["sy"] - p["th"] / p["bh"]) < 1e-12
        assert p["sx"] <= 4.0 + 1e-6 or size_mode == "fixed"
        if p["bw"] <= W:                                                                            # fits: stays inside the image
            assert 0 <= p["x"] and p["x"] + p["bw"] <= W and p["pad"][0] == p["pad"][2] == 0
        if p["bh"] <= H:
            assert 0 <= p["y"] and p["y"] + p["bh"] <= H and p["pad"][1] == p["pad"][3] == 0
        if size_mode == "native (no resize)":
            assert p["sx"] == 1.0 and p["sy"] == 1.0
        if size_mode == "megapixels" and p["sx"] < 4.0 - 1e-6 and min(p["tw"], p["th"]) > 64:
            assert abs(p["tw"] * p["th"] / 2e6 - 1) < 0.25
        # squash check: the crop's aspect equals the model's aspect to within one pixel of the box
        assert abs(p["bw"] / p["bh"] - p["tw"] / p["th"]) <= max(1.5 / p["bh"], 1.5 / p["bw"]) * max(p["bw"] / p["bh"], 1) + 0.05


# ------------------------------------------------------------------ identity: nothing changed -> nothing changed
@pytest.mark.parametrize("H,W", [(200, 300), (333, 517), (64, 64)])
def test_identity_round_trip_is_bit_exact_when_not_resizing(H, W):
    img = make_image(H, W)
    mask = disk_mask(H, W, H // 2, W // 2, min(H, W) // 6)
    st, cropped, cmask, info = ic.crop(img, mask, size_mode="native (no resize)", multiple=16)
    out, alpha, report = ic.stitch(st, cropped)
    assert torch.equal(out, img), info + report                         # every pixel, including the edit area
    assert "no alignment applied" in report


@pytest.mark.parametrize("mode,mp", [("megapixels", 0.05), ("megapixels", 0.3), ("longest_side", 96)])
def test_identity_with_resizing_is_exact_outside_and_close_inside(mode, mp):
    H, W = 400, 600
    img = make_smooth(H, W)
    mask = disk_mask(H, W, 200, 300, 60)
    st, cropped, *_ = ic.crop(img, mask, size_mode=mode, megapixels=mp, longest_side=96, multiple=16)
    assert (st["plan"]["sx"], st["plan"]["sy"]) != (1.0, 1.0)
    out, alpha, report = ic.stitch(st, cropped, align="off", color_match="off")
    outside = alpha == 0
    assert bool(outside.any()) and torch.equal(out[0][outside[0]], img[0][outside[0]])           # untouched = original bits
    assert float((out - img).abs().max()) < 0.01 * float(img.max())                              # smooth content survives the trip
    assert float((out - img).abs().mean()) < 0.002 * float(img.max())


def test_shrinking_then_enlarging_is_smooth_not_blocky():
    """The 'area' filter used to enlarge is plain nearest-neighbour; direction-aware resizing must beat it."""
    H, W = 600, 800
    img = make_smooth(H, W, hdr=4.0)
    mask = disk_mask(H, W, 300, 400, 150)
    st, cropped, *_ = ic.crop(img, mask, size_mode="megapixels", megapixels=0.03, multiple=16)     # strong shrink
    assert st["plan"]["sx"] < 0.5
    out, alpha, _ = ic.stitch(st, cropped, align="off", color_match="off", blend_mode="hard")
    box = ic.extract_box(img, st["plan"])
    good = float((ic.resize_img(cropped, st["plan"]["bh"], st["plan"]["bw"], "bicubic") - box).abs().mean())
    nearest = float((torch.nn.functional.interpolate(cropped.permute(0, 3, 1, 2), size=(st["plan"]["bh"], st["plan"]["bw"]),
                                                     mode="area").permute(0, 2, 3, 1) - box).abs().mean())
    assert good < 0.5 * nearest, (good, nearest)


def test_cropped_values_stay_within_the_source_range_hdr_safe():
    img = make_image(300, 400, hdr=60.0)
    img[0, 100:110, 100:110] = 200.0                                    # a hot highlight
    mask = disk_mask(300, 400, 150, 200, 40)
    for mode in ("megapixels", "native (no resize)"):
        st, cropped, *_ = ic.crop(img, mask, size_mode=mode, megapixels=0.05)
        assert float(cropped.max()) <= float(img.max()) and float(cropped.min()) >= float(img.min())
    out, *_ = ic.stitch(st, cropped)
    assert float(out.max()) <= float(img.max()) + 1e-6 and float(out.min()) >= float(img.min()) - 1e-6   # never clipped to 0..1


# ------------------------------------------------------------------ blend quality
@pytest.mark.parametrize("have_scipy", [True, False])
def test_alpha_is_one_inside_zero_beyond_the_feather_and_monotonic_and_smooth(have_scipy, monkeypatch):
    pytest.importorskip("scipy")
    if not have_scipy:
        monkeypatch.setattr(ic, "_ndi", None)                                               # the blur-based fallback
    H, W = 300, 400
    core = disk_mask(H, W, 150, 200, 50)
    for feather in (8, 24, 48):
        a = ic.build_alpha(core, None, "mask_feather", feather)[0]
        assert bool((a[core[0] > 0] == 1).all())
        d = distance_from_mask(core[0])
        assert bool((a[d > feather + 1] == 0).all())                                       # exact original beyond the reach
        row = a[150, 250:]                                                                  # a line leaving the disk
        assert bool((row[1:] <= row[:-1] + 1e-6).all())                                     # monotonic falloff
        if have_scipy:                                                                       # the blur fallback is only exact, not smooth
            step = float((row[1:] - row[:-1]).abs().max())
            assert step < 1.6 / feather + 1e-3, step                                         # smoothstep: no kink at the edge
            assert float(row[0] - row[1]) < 0.05                                             # zero slope where the mask ends
    hard = ic.build_alpha(core, None, "hard", 24)[0]
    assert torch.equal(hard, core[0])
    soft = torch.full_like(core, 0.4)
    assert float(ic.build_alpha(core, soft, "mask_feather", 8).min()) >= 0.4                # a user's soft mask is respected


def test_everything_beyond_the_feather_is_bit_exact_even_if_the_model_returns_garbage():
    H, W = 500, 700
    img = make_image(H, W)
    mask = disk_mask(H, W, 250, 350, 70)
    st, cropped, *_ = ic.crop(img, mask, size_mode="megapixels", megapixels=0.4, multiple=16)
    garbage = torch.rand_like(cropped) * 90 - 20
    out, alpha, report = ic.stitch(st, garbage, align="off", color_match="off")
    assert bool((alpha[0][distance_from_mask(st["core"][0]) > 40] == 0).all())
    same = out[0] == img[0]
    assert bool(same.all(dim=-1)[alpha[0] == 0].all())
    assert not torch.equal(out, img)


# ------------------------------------------------------------------ the model drifts: alignment and colour
def _drifting_model_result(img, st, shift=(0.37, -0.61), gain=1.06, offset=0.03, seed=5):
    """What an edit model typically does: re-renders the crop slightly shifted and re-graded, with new content inside."""
    p = st["plan"]
    crop = ic.resize_img(ic.extract_box(img, p), p["th"], p["tw"])
    gen = ic.shift_image(crop, -shift[0] * p["sx"], -shift[1] * p["sy"]) * gain + offset
    cm = ic.resize_mask(ic.extract_box_mask(st["core"], p), p["th"], p["tw"])
    g = torch.Generator().manual_seed(seed)
    new = torch.rand(gen.shape, generator=g) * float(img.mean()) * 2
    return gen * (1 - cm.unsqueeze(-1)) + new * cm.unsqueeze(-1)


def _ring_error(out, img, st, inner=3, outer=60):
    d = distance_from_mask(st["core"][0])
    ring = (d > inner) & (d < outer)
    return float((out[0] - img[0]).abs().mean(dim=-1)[ring].mean())


def test_alignment_and_colour_match_remove_the_seam_a_naive_paste_shows():
    H, W = 600, 800
    img = make_image(H, W, hdr=6.0)
    mask = disk_mask(H, W, 300, 400, 90)
    st, cropped, *_ = ic.crop(img, mask, size_mode="megapixels", megapixels=1.0, multiple=16)
    gen = _drifting_model_result(img, st)
    naive, *_ = ic.stitch(st, gen, blend_mode="box_feather", align="off", color_match="off")
    ours, alpha, report = ic.stitch(st, gen, blend_mode="mask_feather", align="translation", color_match="low_frequency")
    assert "aligned the result by" in report and "colour field applied" in report
    e_naive, e_ours = _ring_error(naive, img, st), _ring_error(ours, img, st)
    assert e_ours < 0.35 * e_naive, (e_ours, e_naive, report)
    shift_line = [l for l in report.split("; ") if "aligned" in l][0]
    sx, sy = [float(v) for v in shift_line.split("(")[1].split(")")[0].replace("+", "").split(",")]
    assert abs(sx - 0.37) < 0.08 and abs(sy + 0.61) < 0.08                      # recovered the drift (sub-pixel)


def test_alignment_is_skipped_when_it_would_not_help():
    H, W = 300, 400
    img = make_image(H, W)
    mask = disk_mask(H, W, 150, 200, 40)
    st, cropped, *_ = ic.crop(img, mask, size_mode="native (no resize)")
    noise = torch.rand_like(cropped) * 5                                         # unrelated content: nothing to align to
    out, alpha, report = ic.stitch(st, noise, align="translation", color_match="off")
    assert "no alignment applied" in report


# ------------------------------------------------------------------ robustness
@pytest.mark.parametrize("H,W", SIZES[:6])
def test_any_image_size_with_masks_at_edges_and_corners(H, W):
    img = make_image(H, W, hdr=8.0)
    for (cy, cx, r) in [(0, 0, max(2, min(H, W) // 8)), (H - 1, W - 1, max(2, min(H, W) // 8)), (H // 2, 0, max(2, min(H, W) // 10)),
                        (H // 2, W // 2, max(2, min(H, W) // 3))]:
        mask = disk_mask(H, W, cy, cx, r)
        if not bool(mask.any()):
            continue
        st, cropped, cmask, info = ic.crop(img, mask, size_mode="megapixels", megapixels=0.25, multiple=16)
        assert cropped.shape[1] % 16 == 0 and cropped.shape[2] % 16 == 0 and cmask.shape[1:] == cropped.shape[1:3]
        out, alpha, report = ic.stitch(st, cropped)
        assert out.shape == img.shape and alpha.shape == (1, H, W)
        d = distance_from_mask(st["core"][0])
        assert bool((alpha[0][d > 80] == 0).all())
        assert torch.equal(out[0][alpha[0] == 0], img[0][alpha[0] == 0])
        assert float((out - img).abs().max()) < 0.3 * float(img.max())                    # identity-ish result, no garbage at borders


def test_batch_uses_one_union_box_and_broadcasts_a_single_mask():
    img = make_image(300, 400, B=3)
    st, cropped, cmask, info = ic.crop(img, disk_mask(300, 400, 150, 200, 40), size_mode="native (no resize)")
    assert cropped.shape[0] == 3 and cmask.shape[0] == 3
    out, alpha, _ = ic.stitch(st, cropped)
    assert torch.equal(out, img)
    m = torch.zeros(3, 300, 400)
    m[0, 50:60, 50:60], m[2, 200:260, 300:380] = 1, 1                             # different masks per frame -> union box
    st2, c2, *_ = ic.crop(img, m, size_mode="native (no resize)", context_scale=1.0, context_pixels=0)
    assert st2["plan"]["bbox"] == (50, 50, 380, 260) and torch.equal(ic.stitch(st2, c2)[0], img)


def test_model_result_of_a_different_size_or_with_alpha_channel_is_handled():
    H, W = 500, 700
    img = make_image(H, W)
    mask = disk_mask(H, W, 250, 350, 70)
    st, cropped, *_ = ic.crop(img, mask, size_mode="megapixels", megapixels=0.3, multiple=16)
    th, tw = cropped.shape[1:3]
    odd = ic.resize_img(cropped, th - 8, tw - 16)                                   # e.g. 1880x1056 instead of 1920x1080
    out, alpha, report = ic.stitch(st, odd, align="off", color_match="off")
    assert "resized to the crop size" in report and torch.equal(out[0][alpha[0] == 0], img[0][alpha[0] == 0])
    rgba = torch.cat([cropped, torch.ones(*cropped.shape[:3], 1)], dim=-1)          # Qwen decodes 4 channels
    out2, *_ = ic.stitch(st, rgba, align="off", color_match="off")
    assert out2.shape == img.shape


def test_empty_mask_passes_the_image_through_and_mismatched_mask_size_is_fixed():
    img = make_image(200, 300)
    st, cropped, cmask, info = ic.crop(img, torch.zeros(1, 200, 300))
    assert st["bypass"] and "empty mask" in info
    out, alpha, report = ic.stitch(st, cropped)
    assert torch.equal(out, img) and float(alpha.sum()) == 0 and "bypass" in report
    st2, c2, _, info2 = ic.crop(img, disk_mask(100, 150, 50, 75, 20), size_mode="native (no resize)")      # half-size mask
    assert "mask resized from 150x100 to 300x200" in info2 and not st2["bypass"]


def test_box_larger_than_the_image_is_padded_and_still_exact_outside():
    img = make_image(60, 80)
    mask = torch.zeros(1, 60, 80)
    mask[:, 5:55, 5:75] = 1
    st, cropped, *_ = ic.crop(img, mask, size_mode="fixed", fixed_width=512, fixed_height=512, multiple=16, context_scale=2.0)
    assert any(st["plan"]["pad"])
    out, alpha, report = ic.stitch(st, cropped)
    assert out.shape == img.shape and float((out - img).abs().max()) < 0.5 * float(img.max())


def test_preview_overlay_is_small_and_marks_the_box():
    img = make_image(2000, 3000)
    st, *_ = ic.crop(img, disk_mask(2000, 3000, 1000, 1500, 200))
    pv = ic.preview_overlay(img, st, 512)
    assert max(pv.shape[1:3]) == 512 and pv.shape[-1] == 3 and float(pv[..., 1].max()) == 1.0
    assert ic.preview_overlay(img, {"bypass": True}, 512).shape == pv.shape


def test_hole_filling_without_scipy_matches_scipy_exactly(monkeypatch):
    np = pytest.importorskip("numpy")
    ndi = pytest.importorskip("scipy.ndimage")
    rng = np.random.default_rng(0)
    for _ in range(80):
        H, W = rng.integers(12, 100, 2)
        m = np.zeros((H, W), bool)
        yy, xx = np.ogrid[:H, :W]
        for _ in range(rng.integers(1, 6)):
            y, x, r = rng.integers(0, H), rng.integers(0, W), rng.integers(2, 25)
            m |= (yy - y) ** 2 + (xx - x) ** 2 <= r * r
            if rng.random() < 0.6:
                m &= ~(((yy - y) ** 2 + (xx - x) ** 2) <= rng.integers(0, r) ** 2)
        if rng.random() < 0.3:
            m |= rng.random((H, W)) < 0.05                                  # speckle: diagonal gaps must behave like scipy
        expected = ndi.binary_fill_holes(m)
        monkeypatch.setattr(ic, "_ndi", None)
        assert np.array_equal(ic._fill_holes_one(m), expected)
        monkeypatch.undo()

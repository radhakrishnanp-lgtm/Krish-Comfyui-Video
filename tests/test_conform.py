import importlib.util
import pathlib

import pytest
import torch

_spec = importlib.util.spec_from_file_location(
    "conform", pathlib.Path(__file__).parents[1] / "krish_frame_conform" / "conform.py")
conform = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(conform)


@pytest.mark.parametrize("n,expected", [(1, 1), (2, 9), (9, 9), (10, 17), (97, 97), (100, 105), (121, 121)])
def test_next_valid_frames(n, expected):
    assert conform.next_valid_frames(n) == expected


@pytest.mark.parametrize("n,h,w", [(100, 1080, 1920), (97, 64, 64), (13, 50, 70), (1, 33, 33)])
@pytest.mark.parametrize("fpad", ["repeat_last", "pingpong"])
@pytest.mark.parametrize("anchor", ["center", "top_left"])
def test_roundtrip_is_bit_exact(n, h, w, fpad, anchor):
    if h > 200:  # keep tests light
        h, w = 130, 250
    x = torch.rand(n, h, w, 3)
    padded, info = conform.prepare(x, frame_pad=fpad, spatial_anchor=anchor)
    assert (padded.shape[0] - 1) % 8 == 0 and padded.shape[1] % 32 == 0 and padded.shape[2] % 32 == 0
    out, report = conform.restore(padded, info)
    assert torch.equal(out, x) and report == []


def test_restore_fixes_wrong_frame_count_and_size():
    x = torch.rand(20, 40, 70, 3)
    padded, info = conform.prepare(x)
    bad = padded[:-3, :-5, :-2]  # model dropped frames and cropped
    out, report = conform.restore(bad, info, retime="linear")
    assert out.shape == x.shape and len(report) == 2


def test_mask_composite_keeps_source_exact():
    x = torch.rand(10, 40, 64, 3)
    padded, info = conform.prepare(x)
    gen = torch.rand_like(padded)
    mask = torch.zeros(40, 64)
    mask[:, 32:] = 1
    out, _ = conform.restore(gen, info, source=x, mask=mask)
    assert torch.equal(out[:, :, :32], x[:, :, :32])
    gen_only, _ = conform.restore(gen, info)
    assert torch.allclose(out[:, :, 32:], gen_only[:, :, 32:])


def test_parity():
    a, b = torch.zeros(5, 8, 8, 3), torch.zeros(4, 8, 8, 3)
    assert conform.parity(a, a) == []
    assert conform.parity(a, b)


@pytest.mark.parametrize("n,fps,sr", [(100, 25.0, 48000), (121, 24.0, 44100)])
def test_restore_audio_exact_duration(n, fps, sr):
    x = torch.rand(n, 32, 32, 3)
    padded, info = conform.prepare(x)
    long_audio = {"waveform": torch.rand(1, 2, int(padded.shape[0] / fps * sr)), "sample_rate": sr}
    out = conform.restore_audio(long_audio, info, fps)
    assert out["waveform"].shape[-1] == round(n / fps * sr)
    short = {"waveform": torch.rand(1, 2, 10), "sample_rate": sr}
    assert conform.restore_audio(short, info, fps)["waveform"].shape[-1] == round(n / fps * sr)


def test_prepare_audio_matches_padded_video_then_restores():
    x = torch.rand(100, 32, 32, 3)
    padded, info = conform.prepare(x)
    src = {"waveform": torch.rand(1, 2, 100 * 48000 // 25), "sample_rate": 48000}
    pa = conform.prepare_audio(src, info, 25.0)
    assert pa["waveform"].shape[-1] == round(info.padded_frames / 25.0 * 48000)
    back = conform.restore_audio(pa, info, 25.0)
    assert torch.equal(back["waveform"], src["waveform"])


def test_audio_accepts_2d_waveform():
    x = torch.rand(20, 32, 32, 3)
    padded, info = conform.prepare(x)
    a = {"waveform": torch.rand(2, 1000), "sample_rate": 48000}
    assert conform.prepare_audio(a, info, 24.0)["waveform"].dim() == 3
    assert conform.restore_audio(a, info, 24.0)["waveform"].dim() == 3


def _scene(n=40, size=96, seed=0):
    """Textured background drifting like a handheld camera + a foreground blob; returns (source, matte)."""
    g = torch.Generator().manual_seed(seed)
    bg = torch.rand(size + 40, size + 40, generator=g)
    bg = torch.nn.functional.avg_pool2d(bg[None, None], 3, 1, 1)[0, 0]
    cam = (torch.randn(n, 2, generator=g).cumsum(0) * 0.8).round().long().clamp(-15, 15) + 20
    ph = torch.rand(2, generator=g) * 6.28
    fr = 0.12 + torch.rand(2, generator=g) * 0.25
    tt = torch.arange(n).float()
    pos = torch.stack([48 + 22 * torch.sin(fr[0] * tt + ph[0]), 48 + 22 * torch.sin(fr[1] * tt + ph[1])], 1)
    pos = pos + torch.randn(n, 2, generator=g) * 1.0
    src = torch.zeros(n, size, size, 3)
    matte = torch.zeros(n, size, size, 3)
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    for t in range(n):
        frame = bg[cam[t, 0]:cam[t, 0] + size, cam[t, 1]:cam[t, 1] + size]
        blob = (((yy - pos[t, 0]) ** 2 + (xx - pos[t, 1]) ** 2) < 12 ** 2).float()
        blob = torch.nn.functional.avg_pool2d(blob[None, None], 3, 1, 1)[0, 0]
        src[t] = (frame * (1 - blob) + 0.9 * blob)[..., None]
        matte[t] = blob[..., None]
    return src, matte


@pytest.mark.parametrize("n", [17, 39, 121])
def test_aligned_matte_never_hard_fails(n):
    for seed in range(8):
        src, matte = _scene(n, seed=seed)
        problems, warnings, _ = conform.temporal_check(src, matte, alignment="strict")
        assert problems == [], (n, seed, problems)


@pytest.mark.parametrize("shift", [2, -2, 3])
def test_shifted_matte_is_caught_in_strict_and_only_warned_in_warn_mode(shift):
    caught = 0
    for seed in range(8):
        src, matte = _scene(60, seed=seed)
        late = torch.roll(matte, shift, dims=0)
        strict = conform.temporal_check(src, late, alignment="strict")[0]
        lenient = conform.temporal_check(src, late, alignment="warn")[0]
        assert not any("shifted" in p for p in lenient)
        caught += any("shifted" in p for p in strict)
    assert caught >= 7, caught


def test_alignment_skips_short_and_static_clips():
    src, matte = _scene(6)
    assert conform.temporal_check(src, matte, alignment="strict")[0] == []
    static = torch.full((30, 48, 48, 3), 0.5)
    assert conform.temporal_check(static, static.clone(), alignment="strict")[0] == []


def test_temporal_check_detects_duplicates_dropout_nan_and_count():
    src, matte = _scene(40, seed=1)
    dup = matte.clone()
    dup[10] = dup[9]
    assert any("duplicated" in p for p in conform.temporal_check(src, dup)[0])
    drop = matte.clone()
    drop[20] = 0
    assert any("dropout" in p for p in conform.temporal_check(src, drop)[0])
    nan = matte.clone()
    nan[5, 3, 3, 0] = float("nan")
    assert any("non-finite" in p for p in conform.temporal_check(src, nan)[0])
    assert any("frame count" in p for p in conform.temporal_check(src, matte[:-1])[0])


def test_temporal_check_flags_out_of_range():
    src, matte = _scene(40, seed=2)
    assert any("range" in p for p in conform.temporal_check(src, matte * 2)[0])


@pytest.mark.parametrize("n", [1, 2, 7, 8, 9, 10, 39, 40, 41, 100, 121, 122])
@pytest.mark.parametrize("h,w", [(5, 7), (33, 47), (64, 64), (45, 80), (31, 33)])
def test_any_duration_and_size_roundtrip_through_full_chain(n, h, w):
    x = torch.rand(n, h, w, 3)
    padded, info = conform.prepare(x)
    assert (padded.shape[0] - 1) % 8 == 0 and padded.shape[1] % 32 == 0 and padded.shape[2] % 32 == 0
    model_out = padded.mean(-1, keepdim=True).repeat(1, 1, 1, 3)  # stand-in for the matte the model returns
    out, _ = conform.restore(model_out, info)
    assert out.shape == x.shape
    assert conform.parity(x, out) == []


def test_empty_matte_frames_are_not_flagged_as_duplicates():
    src, matte = _scene(40, seed=3)
    matte[10:14] = 0  # subject leaves frame: several identical all-black frames
    problems = conform.temporal_check(src, matte)[0]
    assert not any("duplicated" in p for p in problems)


def _load_pack():
    import sys
    root = pathlib.Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location("krish_pack", root / "__init__.py",
                                                  submodule_search_locations=[str(root)])
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def test_matte_check_node_never_blocks_in_warn_mode():
    node = _load_pack().NODE_CLASS_MAPPINGS["KrishMatteFrameCheck"]()
    src, matte = _scene(40, seed=5)
    out, report, _ = node.run(src, torch.roll(matte, 3, dims=0), "warn", 4, "strict")
    assert out.shape == matte.shape
    out, report, _ = node.run(src, matte[:-1], "warn", 4, "warn")  # count mismatch: report, don't raise
    assert "FAIL" in report
    out, report, _ = node.run(src, torch.rand(40, 4), "warn", 4, "warn")  # garbage input: still passes through
    assert "could not run" in report
    with pytest.raises(ValueError):
        node.run(src, matte[:-1], "error", 4, "warn")

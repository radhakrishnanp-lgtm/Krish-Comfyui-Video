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


def _moving_square(n=40, size=48, seed=0):
    g = torch.Generator().manual_seed(seed)
    steps = torch.rand(n, generator=g) * 3 + (torch.arange(n) % 5 == 0) * 4  # varied speed -> structured motion
    pos = steps.cumsum(0) % (size - 10)
    x = torch.zeros(n, size, size, 3)
    for t in range(n):
        p = int(pos[t])
        x[t, 8:18, p:p + 10, :] = 1.0
    return x


def test_temporal_check_passes_when_aligned():
    src = _moving_square()
    problems, warnings, info = conform.temporal_check(src, src.clone())
    assert problems == [] and info["best_lag"] == 0


def test_temporal_check_detects_time_shift():
    src = _moving_square()
    shifted = torch.cat([src[:1], src[:1], src[:-2]])  # result lags the source by 1 frame
    problems, _, _ = conform.temporal_check(src, shifted, max_lag=3)
    assert any("shifted" in p or "duplicated" in p for p in problems)
    shifted2 = torch.roll(src, 2, dims=0)
    problems, _, info = conform.temporal_check(src, shifted2, max_lag=4)
    assert any("shifted" in p for p in problems)


def test_temporal_check_detects_duplicates_dropout_nan_and_count():
    src = _moving_square()
    dup = src.clone()
    dup[10] = dup[9]
    assert any("duplicated" in p for p in conform.temporal_check(src, dup)[0])
    drop = src.clone()
    drop[20] = 0
    assert any("dropout" in p for p in conform.temporal_check(src, drop)[0])
    nan = src.clone()
    nan[5, 3, 3, 0] = float("nan")
    assert any("non-finite" in p for p in conform.temporal_check(src, nan)[0])
    assert any("frame count" in p for p in conform.temporal_check(src, src[:-1])[0])


def test_temporal_check_flags_out_of_range():
    src = _moving_square()
    assert any("range" in p for p in conform.temporal_check(src, src * 2)[0])

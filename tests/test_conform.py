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

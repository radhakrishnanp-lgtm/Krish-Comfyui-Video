"""EXR plate -> display encode -> Krish Inpaint Crop -> (stand-in for the edit model) -> Krish Inpaint Stitch -> keep original
outside the edit -> EXR save. The chain used by workflows/Qwen_2.1_Edit_EXR_KrishCrop.json."""
import importlib.util
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

OpenEXR = pytest.importorskip("OpenEXR")
ROOT = pathlib.Path(__file__).parents[1]


def _pack():
    spec = importlib.util.spec_from_file_location("krish_pack_inpaint", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m.NODE_CLASS_MAPPINGS


NODES = _pack()
ic = sys.modules["krish_pack_inpaint.krish_frame_conform.inpaint_core"]
H, W = 384, 512


def write_plate(path):
    g = np.random.default_rng(4)
    n = torch.nn.functional.avg_pool2d(torch.from_numpy(g.random((1, 3, H + 8, W + 8)).astype("float32")), 5, 1, 2)[0, :, 4:-4, 4:-4].numpy()
    ramp = np.linspace(0.2, 1.0, W, dtype="float32")[None, :] * np.linspace(0.3, 1.0, H, dtype="float32")[:, None]
    ch = {c: ((n[i] * 0.6 + ramp) * 9 - 0.03).astype("float16") for i, c in enumerate("RGB")}      # HDR: values up to ~9, some negative
    hdr = {"compression": OpenEXR.PIZ_COMPRESSION, "type": OpenEXR.scanlineimage,
           "chromaticities": (0.7347, 0.2653, 0.0, 1.0, 0.0001, -0.077, 0.32168, 0.33767), "owner": "Pixstone",
           "timeCode": OpenEXR.TimeCode(1, 2, 3, 4, *([0] * 14)), "oiio:ColorSpace": "ACES2065-1"}
    OpenEXR.File(hdr, {k: v.copy() for k, v in ch.items()}).write(str(path))
    return ch


def run_chain(tmp_path, drift=True, size_mode="megapixels", megapixels=0.12):
    ch = write_plate(tmp_path / "plate.1001.exr")
    images, alpha, meta, info, n, w, h, fps = NODES["KrishEXRLoad"]().run(str(tmp_path / "plate.1001.exr"), False, -1, -1, 1, "", 0, 24.0)
    enc = NODES["KrishEXRToDisplay"]().run(images, "sRGB", 0.0, "soft_knee")[0]
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    mask = (((yy - 190) ** 2 + (xx - 260) ** 2) <= 45 ** 2).float()[None]
    st, cropped, cmask, preview, cinfo = NODES["KrishInpaintCrop"]().run(
        enc, mask, 0.01, True, 0, 1.3, 32, size_mode, megapixels, 1536, 1024, 1024, 16, 4.0, "reflect", "bicubic")
    p = st["plan"]
    gen = cropped
    if drift:                                                       # what an edit model does: small shift + re-grade + new content
        gen = ic.shift_image(cropped, -0.4 * p["sx"], 0.25 * p["sy"]) * 1.04 + 0.01
    cm = cmask.unsqueeze(-1)
    gen = gen * (1 - cm) + (cropped * 0.4 + 0.4) * cm
    stitched, blend, report = NODES["KrishInpaintStitch"]().run(st, gen, "mask_feather", "auto", 24, "translation", 12, "low_frequency", 1.0, "bicubic")
    merged, changed, mrep = NODES["KrishEXRKeepOriginalOutsideEdit"]().run(images, enc, stitched, "sRGB", 0.0, "soft_knee", 1e-6)
    ret = NODES["KrishEXRSave"]().run(merged, meta, "replace_rgb", str(tmp_path / "out"), "_edit", "same_as_source", "same_as_source",
                                      "error", "overwrite", "strict")
    return SimpleNamespace(plate=ch, enc=enc, stitcher=st, cropped=cropped, stitched=stitched, blend=blend, report=report,
                           changed=changed, saved=ret)


def read(path):
    f = OpenEXR.File(str(path), separate_channels=True)
    return dict(f.header()), {k: np.array(c.pixels) for k, c in f.channels().items()}


@pytest.mark.parametrize("drift", [True, False])
def test_untouched_pixels_of_the_exr_are_bit_identical_and_the_header_survives(tmp_path, drift):
    r = run_chain(tmp_path, drift)
    outside = (r.blend[0] == 0).numpy()
    assert outside.mean() > 0.8 and (~outside).any()
    assert torch.equal(r.stitched[0][r.blend[0] == 0], r.enc[0][r.blend[0] == 0])            # display space: exact
    hdr, out = read(tmp_path / "out" / "plate_edit.1001.exr")
    for c in "RGB":
        assert np.array_equal(out[c][outside], r.plate[c][outside]), c                      # EXR file: exact, HDR and negatives included
        assert out[c].dtype == np.float16
    assert not np.array_equal(out["R"][~outside], r.plate["R"][~outside])                   # the edit is in there
    assert hdr["owner"] == "Pixstone" and hdr["oiio:ColorSpace"] == "ACES2065-1" and str(hdr["compression"]).endswith("PIZ_COMPRESSION")
    assert "all source attributes identical" in r.saved["result"][1]
    assert float(r.changed.mean()) == pytest.approx(float((r.blend > 0).float().mean()), abs=2e-3)   # merge decodes exactly the blended area


def test_the_model_size_is_a_multiple_of_16_so_the_latent_matches_the_crop(tmp_path):
    r = run_chain(tmp_path)
    h, w = r.cropped.shape[1:3]
    assert h % 16 == 0 and w % 16 == 0
    assert (w // 8 * 8, h // 8 * 8) == (w, h)                                                 # EmptyLatentImage rounds to 8: nothing is rescaled after the crop


def test_the_drift_is_measured_and_corrected_only_when_there_is_one(tmp_path):
    drifting = run_chain(tmp_path / "a", drift=True) if (tmp_path / "a").mkdir() is None else None
    still = run_chain(tmp_path / "b", drift=False) if (tmp_path / "b").mkdir() is None else None
    assert "aligned the result by" in drifting.report and "colour field applied" in drifting.report
    assert "no alignment applied" in still.report


def test_native_mode_changes_nothing_when_the_model_changes_nothing(tmp_path):
    write_plate(tmp_path / "plate.1001.exr")
    images, alpha, meta, *_ = NODES["KrishEXRLoad"]().run(str(tmp_path / "plate.1001.exr"), False, -1, -1, 1, "", 0, 24.0)
    enc = NODES["KrishEXRToDisplay"]().run(images, "sRGB", 0.0, "soft_knee")[0]
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    mask = (((yy - 190) ** 2 + (xx - 260) ** 2) <= 45 ** 2).float()[None]
    st, cropped, *_ = NODES["KrishInpaintCrop"]().run(enc, mask, 0.01, True, 0, 1.3, 32, "native (no resize)", 2.0, 1536, 1024, 1024, 16, 4.0, "reflect", "bicubic")
    stitched, blend, report = NODES["KrishInpaintStitch"]().run(st, cropped, "mask_feather", "auto", 24, "translation", 12, "low_frequency", 1.0, "bicubic")
    assert torch.equal(stitched, enc)                                                         # identity: every pixel, edit area included

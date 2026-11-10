import importlib.util
import pathlib
import struct
import sys
import time

import numpy as np
import pytest
import torch

OpenEXR = pytest.importorskip("OpenEXR")

ROOT = pathlib.Path(__file__).parents[1]
PKG = ROOT / "krish_frame_conform"


def _load_file(name):
    spec = importlib.util.spec_from_file_location(name, PKG / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


exr_io = _load_file("exr_io")
exr_color = _load_file("exr_color")


def _pack():
    spec = importlib.util.spec_from_file_location("krish_pack_exr", ROOT / "__init__.py",
                                                  submodule_search_locations=[str(ROOT)])
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m.NODE_CLASS_MAPPINGS


NODES = _pack()
AP0 = (0.7347, 0.2653, 0.0, 1.0, 0.0001, -0.0770, 0.32168, 0.33767)
H, W = 12, 16


def make_source(folder, frames=(1001, 1002, 1003), compression="PIZ", kind="scanline", multipart=False, seed=0):
    """Production-like frames: HDR + negatives, mixed half/float, AOVs, offset data window, colour tags, timecode."""
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True, exist_ok=True)
    ref = {}
    for fr in frames:
        ch = {
            "R": (rng.random((H, W)) * 40 - 2).astype("float32"),
            "G": (rng.random((H, W)) * 6).astype("float32"),
            "B": (rng.random((H, W)) * 3 - 0.5).astype("float32"),
            "A": rng.random((H, W)).astype("float16"),
            "depth.Z": (rng.random((H, W)) * 100).astype("float32"),
            "cryptomatte00.R": rng.random((H, W)).astype("float32"),
            "objid": rng.integers(0, 1000, (H, W)).astype("uint32"),
        }
        hdr = {
            "compression": getattr(OpenEXR, compression + "_COMPRESSION"),
            "type": OpenEXR.tiledimage if kind == "tiled" else OpenEXR.scanlineimage,
            "dataWindow": ((10, 20), (10 + W - 1, 20 + H - 1)),
            "displayWindow": ((0, 0), (63, 47)),
            "chromaticities": AP0, "whiteLuminance": 100.0, "pixelAspectRatio": 1.0,
            "owner": "Pixstone", "comments": "plate v3", "capDate": "2026:11:06 10:00:00",
            "oiio:ColorSpace": "ACES2065-1", "colorInteropID": "lin_ap0_scene",
            "framesPerSecond": OpenEXR.OpaqueAttribute(struct.pack("<iI", 24, 1), "rational"),
            "timeCode": OpenEXR.TimeCode(1, 2, 3, int(fr) % 24, *([0] * 14)),
            "myCustomFloat": 3.5, "myCustomInt": 7, "myCustomString": "abc",
        }
        if kind == "tiled":
            hdr["tiles"] = OpenEXR.TileDescription()
        ref[fr] = {k: v.copy() for k, v in ch.items()}
        path = folder / f"plate.{fr}.exr"
        if multipart:
            p0 = OpenEXR.Part({**hdr, "name": "beauty"}, {k: v.copy() for k, v in ch.items() if "." not in k})
            p1 = OpenEXR.Part({"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage, "name": "depth",
                               "dataWindow": ((10, 20), (10 + W - 1, 20 + H - 1)), "displayWindow": ((0, 0), (63, 47))},
                              {"Z": ch["depth.Z"].copy()})
            OpenEXR.File([p0, p1]).write(str(path))
        else:
            OpenEXR.File(hdr, {k: v.copy() for k, v in ch.items()}).write(str(path))
    return ref


def read_parts(path):
    f = OpenEXR.File(str(path), separate_channels=True)
    return [(dict(p.header), {n: np.array(c.pixels) for n, c in p.channels.items()}) for p in f.parts]


def load(folder, **kw):
    args = dict(path=str(folder / "plate.1001.exr"), sequence_from_file=True, first_frame=-1, last_frame=-1,
                frame_step=1, layer="", part_index=0, fallback_fps=25.0)
    args.update(kw)
    return NODES["KrishEXRLoad"]().run(**args)


def save(images, meta, out, **kw):
    args = dict(images=images, exr_meta=meta, mode="replace_rgb", output_dir=str(out), name_suffix="",
                pixel_type="same_as_source", compression="same_as_source", size_mismatch="error",
                if_exists="overwrite", verify="strict")
    args.update(kw)
    ret = NODES["KrishEXRSave"]().run(**args)
    return ret["result"] if isinstance(ret, dict) else ret


# ------------------------------------------------------------------ loading
def test_loader_is_unclamped_exact_and_reads_metadata(tmp_path):
    ref = make_source(tmp_path / "src")
    images, alpha, meta, info, n, w, h, fps = load(tmp_path / "src")
    assert (n, w, h) == (3, W, H) and fps == 24.0
    assert images.dtype == torch.float32 and images.shape == (3, H, W, 3)
    assert float(images.max()) > 1 and float(images.min()) < 0
    for i, fr in enumerate((1001, 1002, 1003)):
        assert np.array_equal(images[i, ..., 0].numpy(), ref[fr]["R"])
        assert np.array_equal(images[i, ..., 2].numpy(), ref[fr]["B"])
        assert np.array_equal(alpha[i].numpy(), ref[fr]["A"].astype("float32"))
    assert "lin_ap0_scene" in info and "ACES2065-1" in info and "above 1.0" in info


def test_fps_falls_back_when_file_has_no_frame_rate(tmp_path):
    OpenEXR.File({"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage},
                 {k: np.ones((4, 4), "float32") for k in "RGB"}).write(str(tmp_path / "x.0001.exr"))
    *_, fps = NODES["KrishEXRLoad"]().run(str(tmp_path / "x.0001.exr"), True, -1, -1, 1, "", 0, 29.97)
    assert abs(fps - 29.97) < 1e-6


# ------------------------------------------------------------------ identical round trip
def test_round_trip_is_identical_in_every_channel_and_attribute(tmp_path):
    make_source(tmp_path / "src")
    images, alpha, meta, *_ = load(tmp_path / "src")
    paths, report = save(images, meta, tmp_path / "out")
    assert "all source attributes identical" in report
    for fr in (1001, 1002, 1003):
        (sh, sc), = read_parts(tmp_path / "src" / f"plate.{fr}.exr")
        (oh, oc), = read_parts(tmp_path / "out" / f"plate.{fr}.exr")        # same name AND frame number
        assert exr_io.verify_headers([sh], tmp_path / "out" / f"plate.{fr}.exr") == []
        assert set(sc) == set(oc)
        for name in sc:
            assert sc[name].dtype == oc[name].dtype, name                  # half stays half, uint stays uint
            assert np.array_equal(sc[name], oc[name]), name                # every channel bit-exact, AOVs included
        assert str(oh["compression"]) == str(sh["compression"])
        assert np.array_equal(oh["dataWindow"][0], sh["dataWindow"][0]) and np.array_equal(oh["displayWindow"][1], sh["displayWindow"][1])


@pytest.mark.parametrize("kind,multipart", [("tiled", False), ("scanline", True)])
def test_tiled_and_multipart_round_trip(tmp_path, kind, multipart):
    make_source(tmp_path / "src", kind=kind, multipart=multipart)
    images, alpha, meta, *_ = load(tmp_path / "src")
    save(images, meta, tmp_path / "out")
    src = read_parts(tmp_path / "src" / "plate.1001.exr")
    out = read_parts(tmp_path / "out" / "plate.1001.exr")
    assert len(src) == len(out)
    for (sh, sc), (oh, oc) in zip(src, out):
        assert set(sc) == set(oc)
        assert all(np.array_equal(sc[n], oc[n]) for n in sc)
        assert str(sh["type"]) == str(oh["type"]) and str(sh["compression"]) == str(oh["compression"])
        assert sh.get("name") == oh.get("name")


# ------------------------------------------------------------------ matte modes
def _matte(n=3, seed=5):
    return torch.from_numpy(np.random.default_rng(seed).random((n, H, W)).astype("float32"))


def test_matte_to_alpha_keeps_source_rgb_and_everything_else(tmp_path):
    make_source(tmp_path / "src")
    images, alpha, meta, *_ = load(tmp_path / "src")
    matte = _matte()
    save(images * 0, meta, tmp_path / "out", mode="matte_to_alpha_keep_rgb", matte=matte, name_suffix="_matte")
    (_, sc), = read_parts(tmp_path / "src" / "plate.1002.exr")
    (oh, oc), = read_parts(tmp_path / "out" / "plate_matte.1002.exr")
    for n in sc:
        if n != "A":
            assert np.array_equal(sc[n], oc[n]), n
    assert oc["A"].dtype == np.float16                                     # alpha keeps the source pixel type
    assert np.array_equal(oc["A"], matte[1].numpy().astype("float16"))
    assert oh["owner"] == "Pixstone" and oh["colorInteropID"] == "lin_ap0_scene"


def test_matte_only_alpha_and_gray_rgb(tmp_path):
    make_source(tmp_path / "src")
    images, alpha, meta, *_ = load(tmp_path / "src")
    matte = _matte()
    save(images, meta, tmp_path / "a", mode="matte_only_alpha", matte=matte, keep_other_channels=False)
    (hdr, ch), = read_parts(tmp_path / "a" / "plate.1001.exr")
    assert set(ch) == {"A"} and hdr["chromaticities"] is not None and hdr["comments"] == "plate v3"
    save(images, meta, tmp_path / "b", mode="matte_only_alpha", matte=matte)
    (_, ch), = read_parts(tmp_path / "b" / "plate.1001.exr")
    assert set(ch) == {"A", "depth.Z", "cryptomatte00.R", "objid"}          # AOVs kept, source RGB dropped
    save(images, meta, tmp_path / "c", mode="matte_as_gray_rgb", matte=matte, keep_other_channels=False)
    (_, ch), = read_parts(tmp_path / "c" / "plate.1001.exr")
    assert set(ch) == {"R", "G", "B"} and np.array_equal(ch["R"], ch["B"])
    assert np.array_equal(ch["G"], matte[0].numpy().astype("float32"))


def test_matte_is_derived_from_images_when_not_connected(tmp_path):
    make_source(tmp_path / "src")
    images, alpha, meta, *_ = load(tmp_path / "src")
    gray = torch.from_numpy(np.random.default_rng(2).random((3, H, W, 1)).astype("float32")).repeat(1, 1, 1, 3)
    save(gray, meta, tmp_path / "out", mode="matte_to_alpha_keep_rgb")
    (_, ch), = read_parts(tmp_path / "out" / "plate.1001.exr")
    assert np.allclose(ch["A"].astype("float32"), gray[0, ..., 0].numpy(), atol=1e-3)


# ------------------------------------------------------------------ overrides, safety
def test_pixel_type_and_compression_overrides_keep_attributes(tmp_path):
    make_source(tmp_path / "src")
    images, alpha, meta, *_ = load(tmp_path / "src")
    images = images.clone()
    images[0, 0, 0, 0] = 1e6                                                # would overflow half
    save(images, meta, tmp_path / "out", pixel_type="half", compression="ZIP")
    (sh, _), = read_parts(tmp_path / "src" / "plate.1001.exr")
    (oh, oc), = read_parts(tmp_path / "out" / "plate.1001.exr")
    assert oc["R"].dtype == np.float16 and float(oc["R"][0, 0]) == 65504.0   # clamped, never inf
    assert str(oh["compression"]).endswith("ZIP_COMPRESSION")
    assert oh["myCustomString"] == "abc" and oh["owner"] == sh["owner"]
    save(images, meta, tmp_path / "out2", pixel_type="float", matte=_matte(), mode="matte_to_alpha_keep_rgb")
    (_, oc2), = read_parts(tmp_path / "out2" / "plate.1001.exr")
    assert oc2["A"].dtype == np.float32


def test_size_mismatch_errors_by_default_and_resize_is_opt_in(tmp_path):
    make_source(tmp_path / "src")
    images, alpha, meta, *_ = load(tmp_path / "src")
    big = torch.nn.functional.interpolate(images.permute(0, 3, 1, 2), size=(H * 2, W * 2)).permute(0, 2, 3, 1)
    with pytest.raises(ValueError, match="data window"):
        save(big, meta, tmp_path / "out")
    save(big, meta, tmp_path / "out", size_mismatch="resize")
    (_, oc), = read_parts(tmp_path / "out" / "plate.1001.exr")
    assert oc["R"].shape == (H, W)


def test_never_overwrites_the_source_and_respects_if_exists(tmp_path):
    make_source(tmp_path / "src")
    images, alpha, meta, *_ = load(tmp_path / "src")
    before = (tmp_path / "src" / "plate.1001.exr").read_bytes()
    with pytest.raises(ValueError, match="refusing to overwrite the source"):
        save(images, meta, tmp_path / "src")
    assert (tmp_path / "src" / "plate.1001.exr").read_bytes() == before
    save(images, meta, tmp_path / "out")
    with pytest.raises(FileExistsError):
        save(images, meta, tmp_path / "out", if_exists="error")
    _, report = save(images, meta, tmp_path / "out", if_exists="skip")
    assert "skipped" in report
    assert not list((tmp_path / "out").glob("*.krish_tmp"))                 # no temp files left behind


def test_frame_count_mismatch_is_an_error(tmp_path):
    make_source(tmp_path / "src")
    images, alpha, meta, *_ = load(tmp_path / "src")
    with pytest.raises(ValueError, match="must match one to one"):
        save(images[:2], meta, tmp_path / "out")


def test_header_is_used_when_the_source_file_is_gone(tmp_path):
    make_source(tmp_path / "src")
    images, alpha, meta, *_ = load(tmp_path / "src")
    for p in (tmp_path / "src").glob("*.exr"):
        p.unlink()
    save(images, meta, tmp_path / "out", mode="matte_only_alpha", matte=_matte(), verify="strict")
    (hdr, ch), = read_parts(tmp_path / "out" / "plate.1001.exr")
    assert set(ch) == {"A"} and hdr["oiio:ColorSpace"] == "ACES2065-1" and hdr["owner"] == "Pixstone"


def test_verify_detects_a_changed_attribute(tmp_path):
    make_source(tmp_path / "src")
    images, alpha, meta, *_ = load(tmp_path / "src")
    save(images, meta, tmp_path / "out")
    headers = [dict(h) for h in meta["frames"][0]["headers"]]
    headers[0]["owner"] = "someone else"
    problems = exr_io.verify_headers(headers, tmp_path / "out" / "plate.1001.exr")
    assert any("owner" in p for p in problems)


# ------------------------------------------------------------------ layers
def test_layer_selection_reads_and_writes_the_right_channels(tmp_path):
    make_source(tmp_path / "src")
    images, _, meta, info, *_ = load(tmp_path / "src", layer="depth")
    assert meta["layer_map"]["Y"] == "depth.Z" and float(images.max()) > 50
    save(images * 2, meta, tmp_path / "out", keep_other_channels=True)
    (_, sc), = read_parts(tmp_path / "src" / "plate.1001.exr")
    (_, oc), = read_parts(tmp_path / "out" / "plate.1001.exr")
    assert np.array_equal(oc["depth.Z"], sc["depth.Z"] * 2) and np.array_equal(oc["R"], sc["R"])
    with pytest.raises(ValueError, match="not found"):
        load(tmp_path / "src", layer="nope")


# ------------------------------------------------------------------ sequences
def test_sequence_discovery(tmp_path):
    make_source(tmp_path / "s", frames=(9, 10, 11, 100))
    d = tmp_path / "s"
    nums = lambda recs: [r["frame"] for r in recs]
    assert nums(exr_io.find_sequence(d)) == [9, 10, 11, 100]                # numeric, not alphabetical
    assert nums(exr_io.find_sequence(d / "plate.9.exr", whole_sequence=True)) == [9]   # different padding = other sequence
    make_source(tmp_path / "p", frames=(1001, 1002, 1003, 1004, 1005))
    p = tmp_path / "p"
    assert nums(exr_io.find_sequence(p / "plate.%04d.exr")) == [1001, 1002, 1003, 1004, 1005]
    assert nums(exr_io.find_sequence(p / "plate.####.exr")) == [1001, 1002, 1003, 1004, 1005]
    assert nums(exr_io.find_sequence(p / "plate.1003.exr")) == [1001, 1002, 1003, 1004, 1005]
    assert nums(exr_io.find_sequence(p / "plate.1003.exr", whole_sequence=False)) == [1003]
    assert nums(exr_io.find_sequence(p, first=1002, last=1004)) == [1002, 1003, 1004]
    assert nums(exr_io.find_sequence(p, step=2)) == [1001, 1003, 1005]
    with pytest.raises(FileNotFoundError):
        exr_io.find_sequence(p / "missing.exr")


def test_output_names_keep_frame_numbers_and_support_suffix():
    rec = {"prefix": "shot_010_plate.", "frame_str": "0101", "suffix": ".exr"}
    assert exr_io.output_name(rec) == "shot_010_plate.0101.exr"
    assert exr_io.output_name(rec, "_matte") == "shot_010_plate_matte.0101.exr"
    assert exr_io.output_name({"prefix": "single", "frame_str": None, "suffix": ".exr"}, "_m") == "single_m.exr"


def test_loader_reloads_when_files_change(tmp_path):
    make_source(tmp_path / "src")
    cls = NODES["KrishEXRLoad"]
    a = cls.IS_CHANGED(str(tmp_path / "src"))
    assert a == cls.IS_CHANGED(str(tmp_path / "src"))
    time.sleep(0.02)
    (tmp_path / "src" / "plate.1002.exr").touch()
    assert a != cls.IS_CHANGED(str(tmp_path / "src"))


# ------------------------------------------------------------------ colour
def test_colour_space_is_named_from_chromaticities(tmp_path):
    cases = {"lin_ap0_scene": AP0,
             "lin_rec709_scene": (0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.3290),
             "lin_rec2020_scene": (0.708, 0.292, 0.170, 0.797, 0.131, 0.046, 0.3127, 0.3290)}
    for expect, chrom in cases.items():
        assert expect in exr_io.describe_color({"chromaticities": chrom})
    assert "not set" in exr_io.describe_color({})


@pytest.mark.parametrize("t", ["sRGB", "Rec.709", "gamma 2.2", "gamma 2.4"])
def test_display_transform_round_trips_including_hdr_and_negative_values(t):
    x = torch.tensor([-0.5, -0.01, 0.0, 0.001, 0.18, 0.5, 1.0, 4.0, 37.0])
    back = exr_color.display_to_linear(exr_color._curves(t)[0](x), t)
    assert bool(((back - x).abs() <= 2e-5 * (1 + x.abs())).all())           # curves are exact inverses, no clamping
    y = exr_color.linear_to_display(x, t, 0.0, "clip")
    assert float(y.min()) >= 0 and float(y.max()) <= 1                       # model input is a valid 0..1 image


def test_soft_knee_is_monotonic_continuous_and_keeps_bright_values_apart():
    x = torch.linspace(0, 50, 5001)
    y = exr_color.soft_knee(x)
    assert bool((y[1:] >= y[:-1]).all()) and float(y.max()) < 1.0
    assert torch.allclose(y[x <= 0.8], x[x <= 0.8])
    assert abs(float(exr_color.soft_knee(torch.tensor([0.8000001]))) - 0.8) < 1e-5
    assert float(exr_color.soft_knee(torch.tensor([37.0]))) > float(exr_color.soft_knee(torch.tensor([5.0])))
    d = exr_color.linear_to_display(torch.tensor([1.0, 2.0, 8.0, 37.0]), "sRGB", 0.0, "soft_knee")
    assert bool((d[1:] > d[:-1]).all())                                      # still strictly separated after the curve


def test_exposure_is_applied_in_stops():
    x = torch.full((1, 2, 2, 3), 0.1)
    assert torch.allclose(exr_color.display_to_linear(exr_color.linear_to_display(x, "none", 1.0, "clip"), "none", 1.0), x)


# ------------------------------------------------------------------ errors
def test_missing_openexr_gives_an_actionable_message(monkeypatch):
    monkeypatch.setitem(sys.modules, "OpenEXR", None)
    with pytest.raises(RuntimeError, match="pip install OpenEXR"):
        exr_io._oe()


def test_non_exr_file_is_rejected(tmp_path):
    (tmp_path / "x.0001.exr").write_bytes(b"not an exr")
    with pytest.raises(ValueError, match="not an OpenEXR file"):
        exr_io.read_file(str(tmp_path / "x.0001.exr"))


# ------------------------------------------------------------------ files from another producer (OpenImageIO)
oiio = pytest.importorskip("OpenImageIO")


def _oiio_write(path, comp="zip", dtype=None, subimages=False):
    dtype = dtype or oiio.FLOAT
    rng = np.random.default_rng(11)
    spec = oiio.ImageSpec(W, H, 4, dtype)
    spec.channelnames = ("R", "G", "B", "A")
    spec.attribute("compression", comp)
    spec.attribute("oiio:ColorSpace", "ACEScg")
    spec.attribute("Software", "Nuke 15.1v2")
    spec.attribute("worldToCamera", oiio.TypeMatrix, [float(i) for i in range(16)])
    spec.attribute("myVec3", oiio.TypeVector, (1.0, 2.0, 3.0))
    spec.attribute("myFloatArray", oiio.TypeDesc("float[3]"), (0.1, 0.2, 0.3))
    spec.attribute("myIntArray", oiio.TypeDesc("int[2]"), (5, 6))
    spec.attribute("myStrArray", oiio.TypeDesc("string[2]"), ("a", "b"))
    spec.attribute("nuke/node_hash", "abcdef0123456789")
    spec.attribute("exr/timeCode", oiio.TypeTimeCode, (0x01020304, 0))
    out = oiio.ImageOutput.create(str(path))
    assert out.open(str(path), spec), out.geterror()
    out.write_image((rng.random((H, W, 4)) * 10 - 1).astype("float32"))
    out.close()


@pytest.mark.parametrize("comp,half", [("zip", False), ("piz", True), ("b44", True), ("dwaa:45", False)])
def test_files_from_another_producer_keep_every_attribute(tmp_path, comp, half):
    src = tmp_path / "src"
    src.mkdir()
    _oiio_write(src / "plate.0001.exr", comp, oiio.HALF if half else oiio.FLOAT)
    images, alpha, meta, *_ = NODES["KrishEXRLoad"]().run(str(src / "plate.0001.exr"), True, -1, -1, 1, "", 0, 24.0)
    paths, report = save(images, meta, tmp_path / "out")
    assert "all source attributes identical" in report            # matrices, vectors, arrays, slash keys, timecode ...
    (sh, sc), = read_parts(src / "plate.0001.exr")
    (oh, oc), = read_parts(tmp_path / "out" / "plate.0001.exr")
    assert set(sh) <= set(oh)
    assert np.array_equal(np.asarray(oh["worldToCamera"]), np.asarray(sh["worldToCamera"]))
    assert oh["nuke/node_hash"] == "abcdef0123456789" and oh["myStrArray"] == ["a", "b"]
    assert {n: c.dtype for n, c in sc.items()} == {n: c.dtype for n, c in oc.items()}
    if comp.startswith("dwa"):
        assert oh["dwaCompressionLevel"] == sh["dwaCompressionLevel"]
        assert "WARNING" in report and "second generation of loss" in report
        assert max(float(np.abs(sc[n].astype("float64") - oc[n]).max()) for n in sc) < 0.2   # small, but not zero
    else:
        assert all(np.array_equal(sc[n], oc[n]) for n in sc)      # lossless: bit-exact
        assert "WARNING" not in report


def test_dwa_source_can_be_written_losslessly_by_choosing_zip(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _oiio_write(src / "plate.0001.exr", "dwaa:45")
    images, alpha, meta, *_ = NODES["KrishEXRLoad"]().run(str(src / "plate.0001.exr"), True, -1, -1, 1, "", 0, 24.0)
    _, report = save(images, meta, tmp_path / "out", compression="ZIP")
    assert "WARNING" not in report
    (sh, sc), = read_parts(src / "plate.0001.exr")
    (oh, oc), = read_parts(tmp_path / "out" / "plate.0001.exr")
    assert all(np.array_equal(sc[n], oc[n]) for n in sc)          # decoded source values preserved exactly
    assert str(oh["compression"]).endswith("ZIP_COMPRESSION")


def test_multipart_and_offset_windows_from_another_producer(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    rng = np.random.default_rng(3)
    s0 = oiio.ImageSpec(W, H, 4, oiio.FLOAT)
    s0.channelnames = ("R", "G", "B", "A")
    s0.attribute("name", "beauty")
    s0.attribute("oiio:ColorSpace", "ACEScg")
    s1 = oiio.ImageSpec(W, H, 1, oiio.FLOAT)
    s1.channelnames = ("Z",)
    s1.attribute("name", "depth")
    s1.attribute("compression", "zips")
    p = str(src / "mp.0001.exr")
    out = oiio.ImageOutput.create(p)
    assert out.open(p, [s0, s1])
    beauty = np.ascontiguousarray((rng.random((H, W, 4)) * 5).astype("float32"))
    depth = np.ascontiguousarray((rng.random((H, W, 1)) * 50).astype("float32"))
    out.write_image(beauty)
    out.open(p, s1, "AppendSubimage")
    out.write_image(depth)
    out.close()
    for part in (0, 1):
        images, alpha, meta, *_ = NODES["KrishEXRLoad"]().run(p, True, -1, -1, 1, "", part, 24.0)
        o = tmp_path / f"o{part}"
        save(images, meta, o)
        a = OpenEXR.File(p, separate_channels=True)
        b = OpenEXR.File(str(o / "mp.0001.exr"), separate_channels=True)
        assert len(a.parts) == len(b.parts) == 2
        for pa, pb in zip(a.parts, b.parts):
            assert all(np.array(pa.channels[k].pixels).tobytes() == np.array(pb.channels[k].pixels).tobytes() for k in pa.channels)

    s = oiio.ImageSpec(W, H, 3, oiio.HALF)
    s.channelnames = ("R", "G", "B")
    s.x, s.y, s.full_x, s.full_y, s.full_width, s.full_height = 100, 50, 0, 0, 1920, 1080
    q = str(src / "ow.0001.exr")
    out = oiio.ImageOutput.create(q)
    assert out.open(q, s)
    out.write_image((rng.random((H, W, 3)) * 2).astype("float16"))
    out.close()
    images, alpha, meta, *_ = NODES["KrishEXRLoad"]().run(q, True, -1, -1, 1, "", 0, 24.0)
    save(images, meta, tmp_path / "ow")
    (sh, _), = read_parts(q)
    (oh, _), = read_parts(tmp_path / "ow" / "ow.0001.exr")
    assert np.array_equal(sh["dataWindow"][0], oh["dataWindow"][0]) and tuple(oh["displayWindow"][1]) == (1919, 1079)


def test_end_to_end_matte_pipeline_with_ltx_padding(tmp_path):
    """EXR plates -> display -> pad to LTX grid -> (stand-in model) -> trim -> EXR with source RGB + metadata kept."""
    frames = tuple(range(2001, 2011))                      # 10 frames: not 8n+1
    ref = make_source(tmp_path / "src", frames=frames)     # 16x12 pixels: not a multiple of 32
    images, alpha, meta, info, n, w, h, fps = load(tmp_path / "src", path=str(tmp_path / "src"))
    assert n == 10 and (w, h) == (W, H)

    shown = NODES["KrishEXRToDisplay"]().run(images, "sRGB", 0.0, "soft_knee")[0]
    assert 0 <= float(shown.min()) and float(shown.max()) <= 1

    padded, conform, nframes, pw, ph = NODES["KrishLTXFramePrep"]().run(shown, 8, 32, "repeat_last", "reflect", "center")
    assert nframes == 17 and pw % 32 == 0 and ph % 32 == 0
    model_out = padded.mean(-1, keepdim=True).repeat(1, 1, 1, 3)               # pretend the model returned a matte
    restored, _ = NODES["KrishLTXFrameRestore"]().run(model_out, conform, "nearest", "bilinear")
    assert restored.shape == shown.shape                                      # exact frame count and size again
    NODES["KrishFrameParityCheck"]().run(shown, restored, "error")

    paths, report = save(restored, meta, tmp_path / "out", mode="matte_to_alpha_keep_rgb", name_suffix="_matte",
                         matte=restored[..., 0], keep_other_channels=True)
    assert "all source attributes identical" in report
    outs = sorted((tmp_path / "out").glob("*.exr"))
    assert [o.name for o in outs] == [f"plate_matte.{fr}.exr" for fr in frames]   # same frame numbers
    for fr, o in zip(frames, outs):
        (oh, oc), = read_parts(o)
        assert np.array_equal(oc["R"], ref[fr]["R"]) and np.array_equal(oc["B"], ref[fr]["B"])   # source RGB untouched (HDR kept)
        assert float(oc["R"].max()) > 1
        assert oh["colorInteropID"] == "lin_ap0_scene" and oh["owner"] == "Pixstone"
        assert exr_io.fps_from_header(oh) == 24.0 and oc["A"].shape == (H, W)


# ------------------------------------------------------------------ ACEScct (invertible log encoding)
def test_acescct_matches_the_published_values_and_is_continuous():
    x = torch.tensor([0.0, 0.18, 1.0, 222.86, -0.0069])
    y = exr_color.acescct_encode(x)
    assert abs(float(y[1]) - 0.4135884) < 1e-6                         # mid grey (Academy spec value)
    assert abs(float(y[0]) - 0.0729055) < 1e-6 and abs(float(y[3]) - 1.0) < 1e-3 and abs(float(y[4])) < 2e-4
    cut = torch.tensor([0.0078125 - 1e-7, 0.0078125 + 1e-7])
    assert abs(float(exr_color.acescct_encode(cut)[0] - exr_color.acescct_encode(cut)[1])) < 1e-5


def test_acescct_round_trip_keeps_hdr_highlights():
    x = torch.cat([torch.linspace(-0.006, 0.02, 200), torch.logspace(-2, 2.3, 400)])
    back = exr_color.acescct_decode(exr_color.acescct_encode(x))
    assert bool(((back - x).abs() <= 1e-5 * (1 + x.abs())).all())
    sorted_y = exr_color.acescct_encode(x.sort().values)
    assert bool((sorted_y[1:] >= sorted_y[:-1]).all())                 # monotonic
    name = "ACEScct (log, invertible)"
    hdr = torch.tensor([[[[0.18, 4.0, 37.0]]]])
    enc = exr_color.linear_to_display(hdr, name, 0.0, "clip")
    assert float(enc.max()) <= 1.0 and float(enc.min()) >= 0           # a valid model input
    assert torch.allclose(exr_color.display_to_linear(enc, name), hdr, rtol=1e-4)   # 37.0 comes back, not 1.0


def test_encoding_hint_reads_the_tags():
    assert "scene-linear" in exr_io.suggest_encoding({"colorInteropID": "lin_ap1_scene"})
    assert "display-referred" in exr_io.suggest_encoding({"oiio:ColorSpace": "srgb_rec709_display"})
    assert "no colour tags" in exr_io.suggest_encoding({})


# ------------------------------------------------------------------ keep the original outside a model edit
@pytest.mark.parametrize("t", ["sRGB", "Rec.709", "gamma 2.2"])
def test_soft_knee_round_trip_recovers_hdr_values(t):
    x = torch.cat([torch.linspace(0, 0.8, 50), torch.logspace(-0.1, 1.5, 200)])          # 0 .. ~31
    enc = exr_color.linear_to_display(x, t, 0.0, "soft_knee")
    assert float(enc.min()) >= 0 and float(enc.max()) < 1.0                              # valid model input, nothing clipped
    back = exr_color.display_to_linear(enc, t, 0.0, "soft_knee")
    assert bool(((back - x).abs() <= 2e-3 * (1 + x)).all()), float((back - x).abs().max())


def test_merge_restores_original_bit_exactly_outside_the_edit():
    g = torch.Generator().manual_seed(4)
    orig = torch.rand(1, 40, 60, 3, generator=g) * 12 - 0.01                              # HDR with a few negatives
    enc = exr_color.linear_to_display(orig, "sRGB", 0.0, "soft_knee")
    edited = enc.clone()
    edited[:, 10:20, 15:30] = torch.rand(1, 10, 15, 3, generator=g) * 0.9               # the model "paints" a patch
    out, changed = exr_color.merge_outside_edit(orig, enc, edited, "sRGB", 0.0, "soft_knee")
    inside = torch.zeros(1, 40, 60, dtype=torch.bool)
    inside[:, 10:20, 15:30] = True
    assert torch.equal(out[~inside], orig[~inside])                                       # bit-identical, negatives included
    assert bool((changed.bool() == inside).all())
    assert torch.allclose(out[inside], exr_color.display_to_linear(edited, "sRGB", 0.0, "soft_knee")[inside])
    with pytest.raises(ValueError, match="size mismatch"):
        exr_color.merge_outside_edit(orig, enc, edited[:, :30], "sRGB")


def test_merge_node_and_acescct_variant(tmp_path):
    orig = torch.rand(2, 16, 24, 3) * 40
    node = NODES["KrishEXRKeepOriginalOutsideEdit"]()
    enc = exr_color.linear_to_display(orig, "ACEScct (log, invertible)", 0.0, "clip")
    edited = enc.clone()
    edited[:, 4:8, 4:8] = 0.5
    out, changed, report = node.run(orig, enc, edited, "ACEScct (log, invertible)", 0.0, "clip", 1e-6)
    assert "of pixels were changed" in report and 0 < float(changed.mean()) < 0.1
    outside = ~changed.bool()
    assert torch.equal(out[outside], orig[outside])


def test_from_display_node_can_undo_the_knee():
    x = torch.tensor([[[[0.1, 2.0, 9.0]]]])
    enc = NODES["KrishEXRToDisplay"]().run(x, "sRGB", 0.0, "soft_knee")[0]
    back = NODES["KrishEXRFromDisplay"]().run(enc, "sRGB", 0.0, "soft_knee")[0]
    assert torch.allclose(back, x, rtol=2e-3)
    assert float(NODES["KrishEXRFromDisplay"]().run(enc, "sRGB", 0.0)[0].max()) < 2.0   # default does not undo it


def test_nan_and_inf_pixels_pass_through_bit_exactly(tmp_path):
    """Renderers do write NaN/Inf pixels; the node must neither fix nor spread them, and np.array_equal cannot see them."""
    src = tmp_path / "src"
    src.mkdir()
    rng = np.random.default_rng(9)
    ch = {k: rng.random((H, W)).astype("float32") for k in "RGB"}
    ch["R"][2, 3], ch["G"][4, 5], ch["B"][6, 7], ch["R"][0, 0] = np.nan, np.inf, -np.inf, np.nan
    ch["depth.Z"] = rng.random((H, W)).astype("float32")
    ch["depth.Z"][1, 1] = np.nan
    OpenEXR.File({"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage},
                 {k: v.copy() for k, v in ch.items()}).write(str(src / "n.0001.exr"))
    images, alpha, meta, *_ = NODES["KrishEXRLoad"]().run(str(src / "n.0001.exr"), True, -1, -1, 1, "", 0, 24.0)
    assert bool(torch.isnan(images).any()) and bool(torch.isinf(images).any())          # reported honestly, never clamped
    save(images, meta, tmp_path / "out")
    (_, sc), = read_parts(src / "n.0001.exr")
    (_, oc), = read_parts(tmp_path / "out" / "n.0001.exr")
    assert set(sc) == set(oc)
    assert all(sc[k].tobytes() == oc[k].tobytes() for k in sc)                           # raw bytes, NaN/Inf included


# ------------------------------------------------------------------ Save node on a remote server + download payload
class _FakeFolderPaths:
    def __init__(self, root):
        self.root = root

    def get_output_directory(self):
        return str(self.root / "output")

    def get_input_directory(self):
        return str(self.root / "input")


def _save_raw(images, meta, **kw):
    args = dict(images=images, exr_meta=meta, mode="replace_rgb", output_dir="krish_exr", name_suffix="_qwen",
                pixel_type="same_as_source", compression="same_as_source", size_mismatch="error",
                if_exists="overwrite", verify="strict")
    args.update(kw)
    return NODES["KrishEXRSave"]().run(**args)


def test_save_writes_on_the_server_and_reports_the_exact_location(tmp_path, monkeypatch):
    nodes_mod = sys.modules["krish_pack_exr.krish_frame_conform.exr_nodes"]
    monkeypatch.setattr(nodes_mod, "folder_paths", _FakeFolderPaths(tmp_path / "server"))
    make_source(tmp_path / "src")
    images, alpha, meta, *_ = load(tmp_path / "src")
    ret = _save_raw(images, meta)
    paths, report = ret["result"]
    out_dir = tmp_path / "server" / "output" / "krish_exr"                 # relative output_dir = the SERVER's output folder
    assert sorted(p.name for p in out_dir.glob("*.exr")) == [f"plate_qwen.{fr}.exr" for fr in (1001, 1002, 1003)]
    assert paths.splitlines() == [str(out_dir / f"plate_qwen.{fr}.exr") for fr in (1001, 1002, 1003)]
    assert "SAVED ON THE COMFYUI SERVER: 3 file(s)" in report and str(out_dir) in report and "download button" in report
    entries = ret["ui"]["krish_exr"]                                       # what the browser button uses
    assert [e["name"] for e in entries] == [f"plate_qwen.{fr}.exr" for fr in (1001, 1002, 1003)]
    assert len({e["batch"] for e in entries}) == 1 and all(set(e) == {"id", "name", "size", "batch"} for e in entries)


def test_absolute_output_dir_on_the_server_is_honoured(tmp_path, monkeypatch):
    nodes_mod = sys.modules["krish_pack_exr.krish_frame_conform.exr_nodes"]
    monkeypatch.setattr(nodes_mod, "folder_paths", _FakeFolderPaths(tmp_path / "server"))
    make_source(tmp_path / "src")
    images, alpha, meta, *_ = load(tmp_path / "src")
    show = tmp_path / "mnt" / "shows" / "TD" / "exr"
    _save_raw(images, meta, output_dir=str(show))
    assert len(list(show.glob("*.exr"))) == 3 and not (tmp_path / "server" / "output" / "krish_exr").exists()


def test_save_runs_every_time_even_if_nothing_upstream_changed():
    cls = NODES["KrishEXRSave"]
    a, b = cls.IS_CHANGED(1, 2), cls.IS_CHANGED(1, 2)
    assert a != a and b != b                                               # NaN never equals itself: ComfyUI re-executes the node


def test_unwritable_folder_gives_an_actionable_error(tmp_path, monkeypatch):
    nodes_mod = sys.modules["krish_pack_exr.krish_frame_conform.exr_nodes"]
    monkeypatch.setattr(nodes_mod, "folder_paths", _FakeFolderPaths(tmp_path / "server"))
    make_source(tmp_path / "src")
    images, alpha, meta, *_ = load(tmp_path / "src")
    blocker = tmp_path / "blocked"
    blocker.write_text("a file, not a folder")                              # makedirs under a file fails like a permission error
    with pytest.raises(OSError, match="on the ComfyUI server"):
        _save_raw(images, meta, output_dir=str(blocker / "sub"))


def test_files_written_by_save_are_downloadable_bit_for_bit(tmp_path, monkeypatch):
    import asyncio
    aiohttp = pytest.importorskip("aiohttp")
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer
    nodes_mod = sys.modules["krish_pack_exr.krish_frame_conform.exr_nodes"]
    monkeypatch.setattr(nodes_mod, "folder_paths", _FakeFolderPaths(tmp_path / "server"))
    make_source(tmp_path / "src")
    images, alpha, meta, *_ = load(tmp_path / "src")
    ret = _save_raw(images, meta)
    entries = ret["ui"]["krish_exr"]
    written = ret["result"][0].splitlines()

    async def go():
        routes = web.RouteTableDef()
        nodes_mod.exr_download.add_routes(routes, str(tmp_path))
        app = web.Application()
        app.add_routes(routes)
        async with TestClient(TestServer(app)) as c:
            one = await (await c.get(f"/krish/exr/file?id={entries[0]['id']}")).read()
            zipped = await (await c.get(f"/krish/exr/zip?batch={entries[0]['batch']}")).read()
        return one, zipped

    one, zipped = asyncio.run(go())
    assert one == pathlib.Path(written[0]).read_bytes()
    import io, zipfile
    z = zipfile.ZipFile(io.BytesIO(zipped))
    assert z.namelist() == [pathlib.Path(p).name for p in written]
    assert all(z.read(pathlib.Path(p).name) == pathlib.Path(p).read_bytes() for p in written)

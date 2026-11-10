"""EXR read/write that round-trips everything ComfyUI does not touch.

No ComfyUI imports (unit-testable). OpenEXR is imported lazily, so the rest of the pack loads without it.

Design ("template file"): Save re-opens the ORIGINAL frame and replaces only the channels asked for. Every header
attribute (chromaticities, colour tags, timecode, frame rate, custom attributes, data/display window, compression,
tiling, part names) and every channel that is not replaced (AOVs, cryptomatte, source RGB) is carried over unchanged.
"""
import glob
import os
import re
from fractions import Fraction

import numpy as np

HALF_MAX = 65504.0
COMPRESSIONS = ["NO", "RLE", "ZIPS", "ZIP", "PIZ", "PXR24", "B44", "B44A", "DWAA", "DWAB"]
# Re-encoding already-decoded data with these changes it again (DWA is not idempotent).
LOSSY_NOT_IDEMPOTENT = ("DWAA_COMPRESSION", "DWAB_COMPRESSION")


def _oe():
    try:
        import OpenEXR
    except ImportError as e:  # pragma: no cover - message is tested by monkeypatching
        raise RuntimeError(
            "The Krish EXR nodes need the OpenEXR Python package. Install it into the Python that runs ComfyUI: "
            "python -m pip install OpenEXR") from e
    return OpenEXR


# ------------------------------------------------------------------ sequences
_FRAME_RE = re.compile(r"^(?P<prefix>.*?)(?P<frame>\d+)(?P<suffix>\.[^.\\/]+)$")


def split_frame(name):
    """'plate.0101.exr' -> ('plate.', '0101', '.exr'); None if the name has no frame number."""
    m = _FRAME_RE.match(name)
    return (m["prefix"], m["frame"], m["suffix"]) if m else None


def _record(path):
    parts = split_frame(os.path.basename(path))
    if parts:
        prefix, frame, suffix = parts
        return {"path": path, "prefix": prefix, "frame_str": frame, "frame": int(frame), "suffix": suffix}
    stem, ext = os.path.splitext(os.path.basename(path))
    return {"path": path, "prefix": stem, "frame_str": None, "frame": None, "suffix": ext}


def find_sequence(path, whole_sequence=True, first=-1, last=-1, step=1):
    """Resolve a file, folder or pattern (%04d, ####, @) into frame records, sorted by frame number."""
    path = os.path.expanduser(str(path).strip().strip('"'))
    step = max(1, int(step))
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, "*.exr"))
    elif re.search(r"%0?\d*d|#+|@+", os.path.basename(path)):
        base = os.path.basename(path)

        def tok(m):
            s = m.group(0)
            if s.startswith("%"):
                width = re.search(r"\d+", s)
                return rf"(\d{{{int(width.group())}}})" if width and s[1] == "0" else r"(\d+)"
            return rf"(\d{{{len(s)}}})" if s[0] == "#" else r"(\d+)"

        rx = re.compile("^" + re.sub(r"%0?\d*d|#+|@+", tok, re.escape(base).replace(r"\#", "#").replace(r"\@", "@")) + "$")
        d = os.path.dirname(path) or "."
        files = [os.path.join(d, f) for f in os.listdir(d) if rx.match(f)]
    elif os.path.isfile(path):
        files = [path]
        parts = split_frame(os.path.basename(path)) if whole_sequence else None
        if parts:
            prefix, frame, suffix = parts
            d = os.path.dirname(path) or "."
            files = [os.path.join(d, f) for f in os.listdir(d)
                     if (p := split_frame(f)) and p[0] == prefix and p[2] == suffix and len(p[1]) == len(frame)]
    else:
        raise FileNotFoundError(f"EXR path not found: {path}")
    if not files:
        raise FileNotFoundError(f"no .exr frames found for: {path}")
    recs = sorted((_record(f) for f in files), key=lambda r: (r["frame"] is None, r["frame"] or 0, r["path"]))
    if first >= 0:
        recs = [r for r in recs if r["frame"] is None or r["frame"] >= first]
    if last >= 0:
        recs = [r for r in recs if r["frame"] is None or r["frame"] <= last]
    recs = recs[::step]
    if not recs:
        raise FileNotFoundError(f"no frames left after the first/last/step filter for: {path}")
    return recs


# ------------------------------------------------------------------ reading
def read_file(path, header_only=False):
    oe = _oe()
    path = os.fspath(path)
    if not oe.isOpenExrFile(path):
        raise ValueError(f"not an OpenEXR file: {path}")
    f = oe.File(path, separate_channels=True, header_only=header_only)
    for p in f.parts:
        if str(p.header.get("type", "")).split(".")[-1] in ("deepscanline", "deeptile"):
            raise NotImplementedError(f"deep EXR is not supported: {path}")
    return f


def _clean(parts_header):
    return {k: v for k, v in parts_header.items()}


def part_size(header):
    (x0, y0), (x1, y1) = header["dataWindow"]
    return int(y1 - y0 + 1), int(x1 - x0 + 1)


def list_layers(channel_names):
    """Layer prefixes in a channel list; '' is the base layer."""
    return sorted({n.rsplit(".", 1)[0] if "." in n else "" for n in channel_names})


def resolve_layer(channel_names, layer=""):
    """Map roles to real channel names for one layer.

    Returns {'prefix': 'depth.', 'R': name, 'G': name, 'B': name, 'A': name|None} for colour layers, or
    {'prefix': ..., 'Y': name, 'A': None} for single-channel layers (depth, Y, Z ...).
    """
    names = list(channel_names)
    low = {n.lower(): n for n in names}
    layer = (layer or "").strip()

    def attempt(prefix):
        m = {r: low.get((prefix + r).lower()) for r in "RGBA"}
        if all(m[r] for r in "RGB"):
            return {"prefix": prefix, **m}
        in_layer = [n for n in names if (n.rsplit(".", 1)[0] + "." if "." in n else "") == prefix]
        if len(in_layer) == 1:
            return {"prefix": prefix, "Y": in_layer[0], "A": None}
        if prefix == "" and "y" in low and not any(m.values()):
            return {"prefix": "", "Y": low["y"], "A": low.get("a")}
        return None

    if layer:
        got = attempt(layer + ".")
        if got:
            return got
        raise ValueError(f"layer '{layer}' not found or not RGB / single-channel. Layers: {list_layers(names)}; "
                         f"channels: {sorted(names)}")
    got = attempt("")
    if got:
        return got
    for lay in list_layers(names):
        got = attempt((lay + ".") if lay else "")
        if got:
            return got
    raise ValueError(f"no RGB layer found; channels: {sorted(names)}")


def read_frame(path, layer="", part_index=0, want_pixels=True):
    """Return (rgb [H,W,3] float32, alpha [H,W] float32|None, info dict). Values are never clamped."""
    f = read_file(path, header_only=not want_pixels)
    if part_index >= len(f.parts):
        raise ValueError(f"{path}: part {part_index} requested but the file has {len(f.parts)} part(s)")
    part = f.parts[part_index]
    header = dict(part.header)
    info = {"headers": [dict(p.header) for p in f.parts], "part_index": part_index,
            "part_names": [_safe_name(p) for p in f.parts],
            "channel_names": [list(p.channels.keys()) for p in f.parts]}
    mapping = resolve_layer(part.channels.keys(), layer)
    info["layer_map"] = mapping
    info["size"] = part_size(header)
    if not want_pixels:
        return None, None, info
    ch = part.channels
    if "Y" in mapping:
        g = np.asarray(ch[mapping["Y"]].pixels, dtype=np.float32)
        rgb = np.stack([g, g, g], axis=-1)
    else:
        rgb = np.stack([np.asarray(ch[mapping[r]].pixels, dtype=np.float32) for r in "RGB"], axis=-1)
    alpha = np.asarray(ch[mapping["A"]].pixels, dtype=np.float32) if mapping.get("A") else None
    return rgb, alpha, info


def _safe_name(part):
    try:
        n = part.name() if callable(part.name) else part.name
        return str(n) if n is not None else ""
    except Exception:
        return ""


def fps_from_header(header):
    v = header.get("framesPerSecond")
    if v is None:
        return None
    try:
        if isinstance(v, Fraction):
            return float(v)
        if hasattr(v, "n") and hasattr(v, "d"):
            return float(v.n) / float(v.d)
        a = np.asarray(v).ravel()
        return float(a[0]) / float(a[1]) if a.size == 2 else float(a[0])
    except Exception:
        return None


# ------------------------------------------------------------------ colour description
def describe_color(header):
    """Human-readable colour-space information found in a header. Report only; nothing is converted."""
    oe = _oe()
    lines = []
    chrom = header.get("chromaticities")
    if chrom is not None:
        try:
            cid = oe.chromaticitiesToColorInteropID(tuple(float(x) for x in chrom))
        except Exception:
            cid = None
        lines.append(f"chromaticities: {cid if cid else 'custom primaries'}")
    else:
        lines.append("chromaticities: not set (OpenEXR default = Rec.709 primaries)")
    for key in ("colorInteropID", "oiio:ColorSpace", "colorspace", "ColorSpace", "acesImageContainerFlag",
                "whiteLuminance", "adoptedNeutral"):
        if key in header:
            v = header[key]
            lines.append(f"{key}: {np.asarray(v).tolist() if isinstance(v, np.ndarray) else v}")
    return "; ".join(lines)


def suggest_encoding(header):
    """Is this plate scene-linear or display-referred, judging by its tags? Report only; nothing is converted."""
    tags = " ".join(str(header.get(k, "")) for k in ("colorInteropID", "oiio:ColorSpace", "colorspace", "ColorSpace")).lower()
    if any(m in tags for m in ("lin_", "linear", "acescg", "2065", "scene")):
        return "scene-linear (tags) -> use ACEScct if colour must come back from the model; sRGB is fine for a matte input"
    if any(m in tags for m in ("srgb", "rec709_display", "display", "gamma", "rec.709")):
        return "display-referred (tags) -> use transform 'none' so the model sees the stored values"
    return "no colour tags -> OpenEXR convention is scene-linear; treat as linear"


# ------------------------------------------------------------------ writing
def _to_dtype(arr, dtype):
    a = np.ascontiguousarray(arr)
    if dtype == np.float16:
        return np.clip(a.astype(np.float32), -HALF_MAX, HALF_MAX).astype(np.float16)  # no overflow to inf
    return a.astype(dtype, copy=False)


def _resize(arr, h, w):
    import torch
    import torch.nn.functional as F
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    if t.dim() == 2:
        t = t[None, None]
    else:
        t = t.permute(2, 0, 1)[None]
    t = F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
    return (t[0, 0] if arr.ndim == 2 else t[0].permute(1, 2, 0)).numpy()


def _template(src_path, fallback_headers):
    """[(header dict, {name: Channel})] for every part of the source frame (headers only if the file is gone)."""
    if src_path and os.path.isfile(src_path):
        f = read_file(src_path)
        return [(dict(p.header), dict(p.channels)) for p in f.parts], True, f   # f must outlive the write
    if not fallback_headers:
        raise FileNotFoundError(f"source EXR is gone and no stored header is available: {src_path}")
    return [(dict(h), {}) for h in fallback_headers], False, None


def write_frame(src_path, fallback_headers, out_path, layer_map, part_index, mode, rgb=None, matte=None, *,
                pixel_type="same_as_source", compression="same_as_source", keep_other_channels=True,
                size_mismatch="error", note=""):
    """Write one frame. mode: replace_rgb | matte_to_alpha_keep_rgb | matte_only_alpha | matte_as_gray_rgb."""
    oe = _oe()
    parts, have_src, keepalive = _template(src_path, fallback_headers)
    hdr, chans = parts[part_index]
    H, W = part_size(hdr)

    def fit(a):
        if a is None:
            return None
        if a.shape[0] == H and a.shape[1] == W:
            return a
        if size_mismatch == "resize":
            return _resize(a, H, W)
        raise ValueError(f"image is {a.shape[1]}x{a.shape[0]} but the EXR data window is {W}x{H}. Scale the image to "
                         f"the source size first, or set size_mismatch to 'resize'.")

    rgb, matte = fit(rgb), fit(matte)
    if mode != "replace_rgb" and matte is None:
        raise ValueError(f"mode '{mode}' needs a matte (connect 'matte', or pass an image to derive it from)")
    if mode == "replace_rgb" and rgb is None:
        raise ValueError("mode 'replace_rgb' needs images")

    prefix = layer_map.get("prefix", "")
    color_names = [layer_map[r] for r in "RGB" if layer_map.get(r)] or ([layer_map["Y"]] if layer_map.get("Y") else [])
    a_name = layer_map.get("A") or (prefix + "A")
    layer_names = set(color_names) | ({layer_map["A"]} if layer_map.get("A") else set())
    ref_dtype = np.float32
    for n in color_names:
        if n in chans:
            ref_dtype = chans[n].pixels.dtype
            break

    def dtype_for(name):
        if pixel_type == "half":
            return np.float16
        if pixel_type == "float":
            return np.float32
        return chans[name].pixels.dtype if name in chans else ref_dtype

    new = {}   # channels we set
    drop = set()
    if mode == "replace_rgb":
        if "Y" in layer_map and not layer_map.get("R"):
            new[layer_map["Y"]] = rgb[..., 0]
        else:
            for i, r in enumerate("RGB"):
                new[layer_map[r]] = rgb[..., i if rgb.shape[-1] > i else 0]
        if rgb.shape[-1] >= 4:
            new[a_name] = rgb[..., 3]
        if matte is not None:
            new[a_name] = matte
    elif mode == "matte_to_alpha_keep_rgb":
        new[a_name] = matte
    elif mode == "matte_only_alpha":
        new[a_name] = matte
        drop |= set(color_names)
    elif mode == "matte_as_gray_rgb":
        for r in "RGB":
            new[layer_map.get(r) or (prefix + r)] = matte
        drop |= ({layer_map["A"]} if layer_map.get("A") else set()) | set(color_names) - set(new)
    else:
        raise ValueError(f"unknown mode: {mode}")

    out_chans = {}
    for name, ch in chans.items():
        if name in drop or name in new:
            continue
        if not keep_other_channels and name not in layer_names:
            continue
        out_chans[name] = ch          # untouched: the original Channel object (keeps flags and exact data)
    for name, arr in new.items():
        out_chans[name] = _to_dtype(arr, dtype_for(name))

    new_hdr = {k: v for k, v in hdr.items() if k not in ("channels", "chunkCount")}
    if compression != "same_as_source":
        new_hdr["compression"] = getattr(oe, compression + "_COMPRESSION")
    if note:
        new_hdr["krishProcessing"] = str(note)

    if len(parts) == 1 or not keep_other_channels:
        exr = oe.File(new_hdr, out_chans)
    else:
        built = []
        for i, (h2, c2) in enumerate(parts):
            if i == part_index:
                built.append(oe.Part(new_hdr, out_chans))
            else:
                hh = {k: v for k, v in h2.items() if k not in ("channels", "chunkCount")}
                if compression != "same_as_source":
                    hh["compression"] = getattr(oe, compression + "_COMPRESSION")
                built.append(oe.Part(hh, dict(c2)))
        exr = oe.File(built)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tmp = out_path + ".krish_tmp"
    try:
        exr.write(tmp)
        os.replace(tmp, out_path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    del keepalive  # the source File (which owns the pass-through channels) is only released after the write
    return {"path": out_path, "channels": {n: str(np.asarray(getattr(c, "pixels", c)).dtype) for n, c in out_chans.items()},
            "compression": str(new_hdr.get("compression")).split(".")[-1], "size": (W, H), "template_found": have_src}


# ------------------------------------------------------------------ verification
def _attr_equal(a, b):
    if isinstance(a, float) and isinstance(b, float) and a != a and b != b:
        return True  # NaN == NaN for header comparison
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(_attr_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        x, y = np.asarray(a), np.asarray(b)
        try:
            return np.array_equal(x, y, equal_nan=True)
        except TypeError:
            return np.array_equal(x, y)
    try:
        r = a == b
        return bool(r.all()) if isinstance(r, np.ndarray) else bool(r)
    except Exception:
        return repr(a) == repr(b)


def verify_headers(source_headers, out_path, ignore=()):
    """Compare every source attribute with the written file. Returns a list of mismatch descriptions."""
    f = read_file(out_path, header_only=True)
    problems = []
    skip = {"channels", "chunkCount", *ignore}
    if len(f.parts) != len(source_headers):
        return [f"part count {len(source_headers)} -> {len(f.parts)}"]
    for i, (src, p) in enumerate(zip(source_headers, f.parts)):
        out = dict(p.header)
        for k, v in src.items():
            if k in skip:
                continue
            if k not in out:
                problems.append(f"part {i}: attribute '{k}' missing in output")
            elif not _attr_equal(v, out[k]):
                problems.append(f"part {i}: attribute '{k}' changed: {v!r} -> {out[k]!r}")
    return problems


def output_name(rec, name_suffix="", prefix_override=""):
    """File name for an output frame: same frame number as the source, optional suffix before the frame token."""
    base = prefix_override or rec["prefix"]
    if name_suffix:
        m = re.match(r"^(.*?)([._\-]*)$", base)
        base = m[1] + name_suffix + m[2]
    if rec["frame_str"] is None:
        return base + ".exr"
    return f"{base}{rec['frame_str']}.exr"

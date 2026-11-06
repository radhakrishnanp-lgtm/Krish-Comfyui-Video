"""ComfyUI nodes: EXR in -> processing -> EXR out with the same colour tags, metadata, channels and frame numbers."""
import hashlib
import os

import numpy as np
import torch

from . import exr_color, exr_io

CATEGORY = "Krish/EXR"

try:  # present inside ComfyUI only
    import folder_paths
except Exception:  # pragma: no cover
    folder_paths = None


def _resolve(path):
    p = os.path.expanduser(str(path).strip().strip('"'))
    if folder_paths is not None and not os.path.isabs(p) and not os.path.exists(p):
        cand = os.path.join(folder_paths.get_input_directory(), p)
        if os.path.exists(cand) or "%" in p or "#" in p:
            return cand
    return p


def _out_dir(output_dir):
    base = folder_paths.get_output_directory() if folder_paths is not None else os.getcwd()
    d = os.path.expanduser(str(output_dir).strip()) or "krish_exr"
    return d if os.path.isabs(d) else os.path.join(base, d)


class KrishEXRLoad:
    """Load an EXR file or sequence as unclamped float32 and keep everything needed to write it back identically."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "path": ("STRING", {"default": "", "tooltip": "A frame, a folder, or a pattern like plate.%04d.exr / plate.####.exr"}),
            "sequence_from_file": ("BOOLEAN", {"default": True, "tooltip": "If path is one frame, load its whole numbered sequence."}),
            "first_frame": ("INT", {"default": -1, "min": -1, "max": 10 ** 9, "tooltip": "Frame NUMBER from the file names. -1 = from the start."}),
            "last_frame": ("INT", {"default": -1, "min": -1, "max": 10 ** 9, "tooltip": "Frame NUMBER from the file names. -1 = to the end."}),
            "frame_step": ("INT", {"default": 1, "min": 1, "max": 1000}),
            "layer": ("STRING", {"default": "", "tooltip": "Empty = the main RGB(A) layer. Otherwise a layer prefix, e.g. 'depth' or 'ViewLayer.Combined'."}),
            "part_index": ("INT", {"default": 0, "min": 0, "max": 64, "tooltip": "For multi-part EXRs."}),
            "fallback_fps": ("FLOAT", {"default": 24.0, "min": 0.01, "max": 1000.0, "tooltip": "Used when the file has no framesPerSecond attribute."}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "EXR_META", "STRING", "INT", "INT", "INT", "FLOAT")
    RETURN_NAMES = ("images", "alpha", "exr_meta", "info", "frame_count", "width", "height", "fps")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, path, sequence_from_file=True, first_frame=-1, last_frame=-1, frame_step=1, **kw):
        try:
            recs = exr_io.find_sequence(_resolve(path), sequence_from_file, first_frame, last_frame, frame_step)
            h = hashlib.sha1()
            for r in recs:
                st = os.stat(r["path"])
                h.update(f"{r['path']}|{st.st_mtime_ns}|{st.st_size}".encode())
            return h.hexdigest()
        except Exception:
            return float("nan")

    def run(self, path, sequence_from_file, first_frame, last_frame, frame_step, layer, part_index, fallback_fps):
        recs = exr_io.find_sequence(_resolve(path), sequence_from_file, first_frame, last_frame, frame_step)
        rgbs, alphas, frames, layer_map, size = [], [], [], None, None
        for r in recs:
            rgb, alpha, info = exr_io.read_frame(r["path"], layer, part_index)
            if layer_map is None:
                layer_map, size = info["layer_map"], info["size"]
            elif info["layer_map"] != layer_map or info["size"] != size:
                raise ValueError(f"{r['path']}: layer/size differs from the first frame ({info['size']} vs {size})")
            rgbs.append(rgb)
            alphas.append(alpha)
            frames.append({**r, "headers": info["headers"]})
        H, W = size
        images = torch.from_numpy(np.stack(rgbs))
        have_a = any(a is not None for a in alphas)
        alpha = torch.from_numpy(np.stack([a if a is not None else np.ones((H, W), np.float32) for a in alphas])) \
            if have_a else torch.ones((len(rgbs), H, W), dtype=torch.float32)
        fps = exr_io.fps_from_header(frames[0]["headers"][part_index])
        meta = {"version": 1, "layer": layer, "part_index": part_index, "layer_map": layer_map,
                "frames": frames, "width": W, "height": H, "fps": fps}
        h0 = frames[0]["headers"][part_index]
        comp = str(h0.get("compression")).split(".")[-1]
        names = {v for k, v in layer_map.items() if k != "prefix" and isinstance(v, str)}
        lines = [f"{len(recs)} frame(s) {W}x{H}  first: {recs[0]['path']}",
                 f"layer: '{layer or 'main'}' channels: {sorted(names)}  alpha: {'yes' if have_a else 'no'}",
                 f"compression: {comp}  parts: {len(frames[0]['headers'])}  fps: {fps if fps else f'not in file (using {fallback_fps})'}",
                 f"color: {exr_io.describe_color(h0)}",
                 f"encoding hint: {exr_io.suggest_encoding(h0)}",
                 f"value range: {float(images.min()):.4f} .. {float(images.max()):.4f}  "
                 f"({float((images > 1).float().mean()) * 100:.2f}% of samples above 1.0, "
                 f"{float((images < 0).float().mean()) * 100:.2f}% below 0; nothing is clamped)"]
        return (images, alpha, meta, "\n".join(lines), len(recs), W, H, float(fps) if fps else float(fallback_fps))


SAVE_MODES = ["replace_rgb", "matte_to_alpha_keep_rgb", "matte_only_alpha", "matte_as_gray_rgb"]


class KrishEXRSave:
    """Write EXR frames by re-opening each source frame and replacing only the channels you choose."""

    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE", {"tooltip": "Scene-linear float values (convert display->linear first if needed)."}),
            "exr_meta": ("EXR_META",),
            "mode": (SAVE_MODES, {"tooltip": "replace_rgb: images become R,G,B(,A). matte_to_alpha_keep_rgb: source RGB untouched, "
                                              "matte written to A. matte_only_alpha: file holds only the matte in A. "
                                              "matte_as_gray_rgb: R=G=B=matte."}),
            "output_dir": ("STRING", {"default": "krish_exr", "tooltip": "Absolute, or relative to the ComfyUI output folder."}),
            "name_suffix": ("STRING", {"default": "", "tooltip": "Inserted before the frame number, e.g. _matte -> plate_matte.0101.exr"}),
            "pixel_type": (["same_as_source", "half", "float"], {"tooltip": "Applies to channels this node writes; other channels keep their type."}),
            "compression": (["same_as_source"] + exr_io.COMPRESSIONS,),
            "size_mismatch": (["error", "resize"], {"default": "error"}),
            "if_exists": (["overwrite", "error", "skip"],),
            "verify": (["strict", "report", "off"], {"tooltip": "Re-read every written header and compare each source attribute."}),
        }, "optional": {
            "matte": ("MASK", {"tooltip": "True alpha (not inverted). If missing, the matte is taken from the images."}),
            "keep_other_channels": ("BOOLEAN", {"default": True, "tooltip": "Keep AOVs / other layers / other parts from the source."}),
            "note": ("STRING", {"default": "", "tooltip": "If set, stored in a 'krishProcessing' attribute. Empty = header unchanged."}),
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("paths", "report")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, images, exr_meta, mode, output_dir, name_suffix, pixel_type, compression, size_mismatch, if_exists,
            verify, matte=None, keep_other_channels=True, note=""):
        frames = exr_meta["frames"]
        n = images.shape[0]
        if n != len(frames):
            raise ValueError(f"{n} image(s) but the EXR sequence has {len(frames)} frame(s); they must match one to one")
        if matte is not None and matte.shape[0] not in (1, n):
            raise ValueError(f"matte has {matte.shape[0]} frame(s), expected {n}")
        out_dir = _out_dir(output_dir)
        img = images.detach().cpu().float()
        mt = matte.detach().cpu().float() if matte is not None else None
        paths, lines, problems_all = [], [], []
        for i, rec in enumerate(frames):
            out_path = os.path.join(out_dir, exr_io.output_name(rec, name_suffix))
            src = rec["path"]
            if os.path.exists(src) and os.path.realpath(src) == os.path.realpath(out_path):
                raise ValueError(f"refusing to overwrite the source frame: {src}. Choose another output_dir or name_suffix.")
            if os.path.exists(out_path) and if_exists != "overwrite":
                if if_exists == "error":
                    raise FileExistsError(out_path)
                paths.append(out_path)
                lines.append(f"skipped (exists): {out_path}")
                continue
            m = None
            if mode != "replace_rgb":
                if mt is not None:
                    m = mt[min(i, mt.shape[0] - 1)].numpy()
                else:
                    m = (img[i][..., :3].mean(-1) if img.shape[-1] >= 3 else img[i][..., 0]).numpy()
            elif mt is not None:
                m = mt[min(i, mt.shape[0] - 1)].numpy()
            res = exr_io.write_frame(
                src, [h for h in rec["headers"]], out_path, exr_meta["layer_map"], exr_meta["part_index"], mode,
                rgb=img[i].numpy() if mode == "replace_rgb" else None, matte=m,
                pixel_type=pixel_type, compression=compression, keep_other_channels=keep_other_channels,
                size_mismatch=size_mismatch, note=note)
            paths.append(out_path)
            if verify != "off":
                ignore = {"compression"} if compression != "same_as_source" else set()
                if note:
                    ignore.add("krishProcessing")
                bad = exr_io.verify_headers(rec["headers"], out_path, ignore)
                problems_all += [f"{os.path.basename(out_path)}: {b}" for b in bad]
            if i == 0:
                lines.append(f"first frame: {out_path}  channels: {res['channels']}  compression: {res['compression']}  "
                             f"source template: {'yes' if res['template_found'] else 'NO (header from memory)'}")
                if res["compression"] in exr_io.LOSSY_NOT_IDEMPOTENT:
                    lines.append(f"WARNING: output compression is {res['compression']} (lossy). Any channel this node does not "
                                 f"change is re-compressed from its decoded values, a second generation of loss. Choose ZIP or "
                                 f"PIZ in 'compression' to keep untouched channels bit-identical to the decoded source.")
        lines.insert(0, f"wrote {len([p for p in paths])} frame(s) to {out_dir}  mode: {mode}")
        if verify != "off":
            lines.append("header check: " + ("all source attributes identical in every output frame" if not problems_all
                                              else f"{len(problems_all)} difference(s)"))
            lines += problems_all[:20]
        report = "\n".join(lines)
        print("[KrishEXR] " + report.replace("\n", "\n[KrishEXR] "))
        if problems_all and verify == "strict":
            raise ValueError("EXR header verification failed (files were written; set verify to 'report' to accept):\n" + report)
        return ("\n".join(paths), report)


class KrishEXRToDisplay:
    """Scene-linear EXR values -> display-referred 0..1 for models (LTX etc.). Does not change the EXR or its header."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "transform": (exr_color.TRANSFORMS, {"default": "sRGB"}),
            "exposure_stops": ("FLOAT", {"default": 0.0, "min": -16.0, "max": 16.0, "step": 0.1}),
            "highlights": (exr_color.HIGHLIGHTS, {"tooltip": "clip: hard 0..1. soft_knee: compress values above 0.8 smoothly before the curve."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, images, transform, exposure_stops, highlights):
        return (exr_color.linear_to_display(images, transform, exposure_stops, highlights),)


class KrishEXRFromDisplay:
    """Display-referred -> scene-linear, unclamped. Inverse of Linear -> Display (clip cannot be undone)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "transform": (exr_color.TRANSFORMS, {"default": "sRGB"}),
            "exposure_stops": ("FLOAT", {"default": 0.0, "min": -16.0, "max": 16.0, "step": 0.1}),
            "highlights": (exr_color.HIGHLIGHTS, {"tooltip": "soft_knee: also undo the soft knee applied by Linear -> Display."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, images, transform, exposure_stops, highlights="clip"):
        return (exr_color.display_to_linear(images, transform, exposure_stops, highlights),)


class KrishEXRKeepOriginalOutsideEdit:
    """Decode a model's edit back to scene-linear, restoring the ORIGINAL linear pixels everywhere the model left alone."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "original_linear": ("IMAGE", {"tooltip": "The EXR plate as loaded (scene-linear)."}),
            "original_encoded": ("IMAGE", {"tooltip": "The exact image that went into the model workflow (output of Linear -> Display)."}),
            "edited_encoded": ("IMAGE", {"tooltip": "The model workflow result at the same size (e.g. after the inpaint stitch)."}),
            "transform": (exr_color.TRANSFORMS, {"default": "sRGB", "tooltip": "Must match Linear -> Display."}),
            "exposure_stops": ("FLOAT", {"default": 0.0, "min": -16.0, "max": 16.0, "step": 0.1, "tooltip": "Must match Linear -> Display."}),
            "highlights": (exr_color.HIGHLIGHTS, {"tooltip": "Must match Linear -> Display. soft_knee undoes the knee exactly."}),
            "tolerance": ("FLOAT", {"default": 1e-6, "min": 0.0, "max": 0.01, "step": 1e-6, "tooltip": "Pixels that differ from the input by less than this count as unchanged."}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("images", "changed_mask", "report")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, original_linear, original_encoded, edited_encoded, transform, exposure_stops, highlights, tolerance):
        out, changed = exr_color.merge_outside_edit(original_linear, original_encoded, edited_encoded, transform,
                                                    exposure_stops, highlights, tolerance)
        pct = float(changed.mean()) * 100
        report = (f"{pct:.2f}% of pixels were changed by the model and decoded; the other {100 - pct:.2f}% are the original "
                  f"linear pixels, untouched. Output range {float(out.min()):.4f} .. {float(out.max()):.4f}")
        print("[KrishEXR] " + report)
        return (out, changed, report)


NODE_CLASS_MAPPINGS = {
    "KrishEXRLoad": KrishEXRLoad,
    "KrishEXRSave": KrishEXRSave,
    "KrishEXRToDisplay": KrishEXRToDisplay,
    "KrishEXRFromDisplay": KrishEXRFromDisplay,
    "KrishEXRKeepOriginalOutsideEdit": KrishEXRKeepOriginalOutsideEdit,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "KrishEXRLoad": "Krish EXR Load (float, sequence, metadata)",
    "KrishEXRSave": "Krish EXR Save (same details as source)",
    "KrishEXRToDisplay": "Krish EXR Linear -> Display (for models)",
    "KrishEXRFromDisplay": "Krish EXR Display -> Linear",
    "KrishEXRKeepOriginalOutsideEdit": "Krish EXR Keep Original Outside Edit (decode + merge)",
}

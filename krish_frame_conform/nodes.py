from . import conform

CATEGORY = "Krish/Video"


class KrishLTXFramePrep:
    """Pad any video to LTX's grid (8n+1 frames, size multiple of 32) without resampling."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_stride": ("INT", {"default": 8, "min": 1, "max": 64,
                                         "tooltip": "Temporal VAE stride. Valid counts are stride*k+1."}),
                "size_multiple": ("INT", {"default": 32, "min": 1, "max": 128}),
                "frame_pad": (["repeat_last", "pingpong"],),
                "spatial_pad": (["reflect", "replicate"],),
                "spatial_anchor": (["center", "top_left"],),
            }
        }

    RETURN_TYPES = ("IMAGE", "KRISH_CONFORM", "INT", "INT", "INT")
    RETURN_NAMES = ("images", "conform_info", "num_frames", "width", "height")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, images, frame_stride, size_multiple, frame_pad, spatial_pad, spatial_anchor):
        out, info = conform.prepare(images, frame_stride, size_multiple,
                                    frame_pad, spatial_pad, spatial_anchor)
        print(f"[KrishLTX] prep: {info.orig_frames}f {info.orig_w}x{info.orig_h} -> "
              f"{info.padded_frames}f {info.padded_w}x{info.padded_h}")
        return (out, info, info.padded_frames, info.padded_w, info.padded_h)


class KrishLTXFrameRestore:
    """Trim LTX output back to exactly the original frame count and size."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generated": ("IMAGE",),
                "conform_info": ("KRISH_CONFORM",),
                "retime": (["nearest", "linear"],),
                "resize_mode": (["bilinear", "bicubic", "nearest"],),
            },
            "optional": {
                "source_images": ("IMAGE", {"tooltip": "Original frames, for mask compositing."}),
                "mask": ("MASK", {"tooltip": "1 = use generated, 0 = keep source pixel-exact."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "report")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, generated, conform_info, retime, resize_mode, source_images=None, mask=None):
        out, report = conform.restore(generated, conform_info, retime, resize_mode,
                                      source_images, mask)
        text = "; ".join(report) or "exact match, only padding trimmed"
        print(f"[KrishLTX] restore: {text}")
        return (out, text)


class KrishFrameParityCheck:
    """Fail loudly if frame count/size differ between source and result."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"source": ("IMAGE",), "result": ("IMAGE",),
                             "on_mismatch": (["error", "warn"],)}}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("result", "report")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, source, result, on_mismatch):
        problems = conform.parity(source, result)
        if not problems:
            return (result, f"OK: {result.shape[0]} frames {result.shape[2]}x{result.shape[1]}")
        msg = "PARITY MISMATCH: " + "; ".join(problems)
        if on_mismatch == "error":
            raise ValueError(msg)
        print(f"[KrishLTX] {msg}")
        return (result, msg)


class KrishMatteFrameCheck:
    """Frame-to-frame QC of a generated matte vs its source: count, NaN/range, time shift, duplicates, dropouts."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "source": ("IMAGE",), "result": ("IMAGE",),
            "on_fail": (["warn", "error"],),
            "max_lag": ("INT", {"default": 4, "min": 0, "max": 24, "tooltip": "Frames of shift to test for."}),
            "alignment": (["warn", "strict", "off"], {"tooltip": "warn: a detected time shift is reported but does "
                           "not stop the run. strict: it stops the run. off: skip the test."}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("result", "report", "flagged_frames")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, source, result, on_fail, max_lag, alignment="warn"):
        try:
            problems, warnings, info = conform.temporal_check(source, result, max_lag=max_lag, alignment=alignment)
        except Exception as e:  # a QC report must never be the reason a render is lost
            if on_fail == "error":
                raise
            msg = f"matte check could not run ({type(e).__name__}: {e}); frames passed through untouched"
            print("[KrishLTX] " + msg)
            return (result, msg, msg)
        lines = [f"frames: {result.shape[0]}  size: {result.shape[2]}x{result.shape[1]}"]
        if "best_lag" in info:
            lines.append(f"alignment: strongest alternative lag {info['best_lag']:+d} "
                         f"(gain {info['mean_gain']:+.3f}, better in {info['frac_better']:.0%} of frames), "
                         f"corr@0 {info['corr_at_0']}")
        lines += [f"FAIL: {p}" for p in problems] + [f"WARN: {w}" for w in warnings]
        if not problems:
            lines.insert(1, "OK: all frame-to-frame checks passed")
        report = "\n".join(lines)
        flagged = "; ".join(problems + warnings)
        print("[KrishLTX] matte check\n" + report)
        if problems and on_fail == "error":
            raise ValueError("MATTE FRAME CHECK FAILED\n" + report)
        return (result, report, flagged)


class KrishLTXAudioPrep:
    """Zero-pad source audio to the padded video length so audio and video latents line up."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"audio": ("AUDIO",), "conform_info": ("KRISH_CONFORM",),
                             "fps": ("FLOAT", {"default": 24.0, "min": 0.01, "max": 1000.0, "forceInput": True})}}

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, audio, conform_info, fps):
        return (conform.prepare_audio(audio, conform_info, fps),)


class KrishLTXAudioRestore:
    """Trim LTX audio (generated for the padded length) to exactly the original duration."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"audio": ("AUDIO",), "conform_info": ("KRISH_CONFORM",),
                             "fps": ("FLOAT", {"default": 24.0, "min": 0.01, "max": 1000.0, "forceInput": True})}}

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, audio, conform_info, fps):
        return (conform.restore_audio(audio, conform_info, fps),)


NODE_CLASS_MAPPINGS = {
    "KrishLTXFramePrep": KrishLTXFramePrep,
    "KrishLTXFrameRestore": KrishLTXFrameRestore,
    "KrishFrameParityCheck": KrishFrameParityCheck,
    "KrishLTXAudioRestore": KrishLTXAudioRestore,
    "KrishLTXAudioPrep": KrishLTXAudioPrep,
    "KrishMatteFrameCheck": KrishMatteFrameCheck,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "KrishLTXFramePrep": "Krish LTX Frame Prep (pad to grid)",
    "KrishLTXFrameRestore": "Krish LTX Frame Restore (exact N frames)",
    "KrishFrameParityCheck": "Krish Frame Parity Check",
    "KrishLTXAudioRestore": "Krish LTX Audio Restore (exact duration)",
    "KrishLTXAudioPrep": "Krish LTX Audio Prep (pad to video)",
    "KrishMatteFrameCheck": "Krish Matte Frame-to-Frame Check",
}

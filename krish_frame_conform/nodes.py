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


class KrishLTXAudioPrep:
    """Zero-pad source audio to the padded video length so audio and video latents line up."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"audio": ("AUDIO",), "conform_info": ("KRISH_CONFORM",),
                             "fps": ("FLOAT", {"default": 24.0, "min": 0.01, "max": 1000.0})}}

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
                             "fps": ("FLOAT", {"default": 24.0, "min": 0.01, "max": 1000.0})}}

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
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "KrishLTXFramePrep": "Krish LTX Frame Prep (pad to grid)",
    "KrishLTXFrameRestore": "Krish LTX Frame Restore (exact N frames)",
    "KrishFrameParityCheck": "Krish Frame Parity Check",
    "KrishLTXAudioRestore": "Krish LTX Audio Restore (exact duration)",
    "KrishLTXAudioPrep": "Krish LTX Audio Prep (pad to video)",
}

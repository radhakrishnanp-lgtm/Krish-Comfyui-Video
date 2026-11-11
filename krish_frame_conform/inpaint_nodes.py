"""ComfyUI nodes: Krish Inpaint Crop / Stitch. See inpaint_core.py for the algorithm."""
import torch

from . import inpaint_core as ic

CATEGORY = "Krish/Inpaint"


class KrishInpaintCrop:
    """Cut the edit region (plus context) out of a big image for a model, remembering the exact geometry."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "mask": ("MASK", {"tooltip": "White = the area to edit. Should be the same size as the image (it is resized if not)."}),
            "mask_threshold": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Mask values above this count as edit area, so even very light greys are detected."}),
            "fill_holes": ("BOOLEAN", {"default": True, "tooltip": "Fill holes completely enclosed by the mask."}),
            "mask_grow": ("INT", {"default": 0, "min": -512, "max": 512, "tooltip": "Grow (positive) or shrink (negative) the mask in pixels before cropping."}),
            "context_scale": ("FLOAT", {"default": 1.3, "min": 1.0, "max": 8.0, "step": 0.05, "tooltip": "Crop box = mask box x this. More surroundings give the model more context."}),
            "context_pixels": ("INT", {"default": 32, "min": 0, "max": 4096, "tooltip": "Minimum extra pixels around the mask box (room for the blend)."}),
            "size_mode": (ic.SIZE_MODES, {"tooltip": "megapixels: scale the crop to about that many megapixels. longest_side: longest side in pixels. fixed: exact width x height. native: no resize at all (exact pixels)."}),
            "megapixels": ("FLOAT", {"default": 2.0, "min": 0.05, "max": 64.0, "step": 0.05}),
            "longest_side": ("INT", {"default": 1536, "min": 64, "max": 16384, "step": 8}),
            "fixed_width": ("INT", {"default": 1024, "min": 8, "max": 16384, "step": 8}),
            "fixed_height": ("INT", {"default": 1024, "min": 8, "max": 16384, "step": 8}),
            "size_multiple": ("INT", {"default": 16, "min": 8, "max": 256, "step": 8, "tooltip": "Model size is a multiple of this (16 for Qwen / Flux)."}),
            "max_upscale": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 16.0, "step": 0.5, "tooltip": "Largest enlargement for small crops."}),
            "edge_pad": (["reflect", "replicate"], {"tooltip": "How to fill the part of the crop box that leaves the image."}),
            "resample": (ic.RESAMPLE, {"tooltip": "Filter used to resize. Shrinking is always antialiased."}),
        }}

    RETURN_TYPES = ("KRISH_STITCHER", "IMAGE", "MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("stitcher", "cropped_image", "cropped_mask", "preview", "info")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, image, mask, mask_threshold, fill_holes, mask_grow, context_scale, context_pixels, size_mode, megapixels,
            longest_side, fixed_width, fixed_height, size_multiple, max_upscale, edge_pad, resample):
        st, cropped, cmask, info = ic.crop(
            image, mask, threshold=mask_threshold, fill_holes=fill_holes, grow=mask_grow, edge_pad=edge_pad,
            resample=resample, context_scale=context_scale, context_pixels=context_pixels, size_mode=size_mode,
            megapixels=megapixels, longest_side=longest_side, fixed_width=fixed_width, fixed_height=fixed_height,
            multiple=size_multiple, max_upscale=max_upscale)
        print("[KrishInpaint] " + info)
        return (st, cropped, cmask, ic.preview_overlay(st["original"], st), info)


class KrishInpaintStitch:
    """Put the model's result back: original pixels outside the edit, a seamless blend at its border."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "stitcher": ("KRISH_STITCHER",),
            "inpainted_image": ("IMAGE", {"tooltip": "The model result for the cropped image (any size; resized to the crop size)."}),
            "blend_mode": (["mask_feather", "hard", "box_feather"], {"tooltip": "mask_feather: follows the mask shape, exactly 0 outside the feather. hard: mask only. box_feather: fades at the crop box edge (the whole box is replaced)."}),
            "feather_mode": (["auto", "pixels"],),
            "feather_pixels": ("INT", {"default": 24, "min": 0, "max": 512, "tooltip": "Used when feather_mode is pixels."}),
            "align": (["affine", "translation", "off"], {"tooltip": "Measure and remove the small shift, zoom and rotation the model introduced, using the untouched surroundings (sub-pixel). translation = shift only."}),
            "max_shift_px": ("INT", {"default": 64, "min": 1, "max": 512, "tooltip": "Largest correction applied, in original-image pixels at the crop box corners. Edit models can drift 10-30 px on a large plate."}),
            "color_match": (["low_frequency", "off"], {"tooltip": "Smooth colour/brightness correction learned on the untouched surroundings and extended into the edit."}),
            "color_amount": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "resample": (ic.RESAMPLE,),
            "protect_letterbox": ("BOOLEAN", {"default": True, "tooltip": "Detect flat black letterbox / pillarbox bars and keep them bit-exact; the blend is held back where the model moved the bar edge (a black wedge bleeding into the picture)."}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "blend_mask", "report")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, stitcher, inpainted_image, blend_mode, feather_mode, feather_pixels, align, max_shift_px, color_match,
            color_amount, resample, protect_letterbox=True):
        out, alpha, report = ic.stitch(
            stitcher, inpainted_image, blend_mode=blend_mode, feather_mode=feather_mode, feather_pixels=feather_pixels,
            align=align, max_shift_px=max_shift_px, color_match=color_match, color_amount=color_amount, resample=resample,
            protect_letterbox=protect_letterbox)
        print("[KrishInpaint] " + report)
        return (out, alpha, report)


NODE_CLASS_MAPPINGS = {"KrishInpaintCrop": KrishInpaintCrop, "KrishInpaintStitch": KrishInpaintStitch}
NODE_DISPLAY_NAME_MAPPINGS = {"KrishInpaintCrop": "Krish Inpaint Crop", "KrishInpaintStitch": "Krish Inpaint Stitch"}

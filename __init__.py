# ComfyUI loads this folder as a package from custom_nodes/. The guard keeps pytest, which
# imports this file outside a package, from failing on the relative import.
if __package__:
    from .krish_frame_conform.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

    # EXR nodes are optional: if anything about them fails to import, the LTX nodes above are unaffected.
    try:
        from .krish_frame_conform import exr_nodes as _exr

        NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **_exr.NODE_CLASS_MAPPINGS}
        NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **_exr.NODE_DISPLAY_NAME_MAPPINGS}
    except Exception as _e:  # pragma: no cover
        print(f"[Krish] EXR nodes disabled: {_e}")

    # Inpaint crop/stitch nodes: pure torch, optional like the EXR nodes.
    try:
        from .krish_frame_conform import inpaint_nodes as _inp

        NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **_inp.NODE_CLASS_MAPPINGS}
        NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **_inp.NODE_DISPLAY_NAME_MAPPINGS}
    except Exception as _e:  # pragma: no cover
        print(f"[Krish] inpaint nodes disabled: {_e}")

    WEB_DIRECTORY = "./web"  # krish_exr_upload.js: the upload button on Krish EXR Load
    KRISH_PACK_VERSION = "0.6.2"
    print(f"[Krish] Krish-Comfyui-Video {KRISH_PACK_VERSION} loaded: {len(NODE_CLASS_MAPPINGS)} nodes "
          f"(EXR nodes: {'yes' if 'KrishEXRLoad' in NODE_CLASS_MAPPINGS else 'NO'}, "
          f"merge node: {'yes' if 'KrishEXRKeepOriginalOutsideEdit' in NODE_CLASS_MAPPINGS else 'NO'}, "
          f"download endpoint: {'yes' if _exr.exr_download._INSTALLED else 'no'}, "
          f"inpaint nodes: {'yes' if 'KrishInpaintCrop' in NODE_CLASS_MAPPINGS else 'NO'})")

    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

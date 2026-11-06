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

    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

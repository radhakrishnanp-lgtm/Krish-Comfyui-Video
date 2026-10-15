# ComfyUI loads this folder as a package from custom_nodes/. The guard keeps pytest, which
# imports this file outside a package, from failing on the relative import.
if __package__:
    from .krish_frame_conform.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

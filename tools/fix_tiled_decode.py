"""Replace VAE Decode (Tiled) with a full VAE Decode in an LTX workflow. Nothing else is touched.

Usage: python3 tools/fix_tiled_decode.py <original.json> <patched.json>

Why: VAEDecodeTiled with temporal_size 64 decodes the clip in separate time chunks. The LTX video VAE compresses time 8:1
and is causal, so one 64-frame tile = 8 latent frames = 57 pixel frames (0..56). Neighbouring tiles are blended where they
overlap and do not line up exactly, which softens fine detail (hair edges in a matte) around frame ~56. A full decode has no
chunk boundaries. (ComfyUI's VAE Decode falls back to tiling by itself only if the GPU runs out of memory.)
"""
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
d = json.load(open(src))
tiled = [n for n in d["nodes"] if n["type"] == "VAEDecodeTiled"]
assert tiled, "no VAEDecodeTiled node found"
for n in tiled:
    n["type"] = "VAEDecode"
    n["properties"]["Node name for S&R"] = "VAEDecode"
    n["widgets_values"] = []
    n.pop("widgets_values_named", None)
    n["size"] = [270, 46]
    n["title"] = "VAE Decode (full, no tiling)"
    n["color"], n["bgcolor"] = "#2a4a3a", "#1f3a2c"
json.dump(d, open(dst, "w"), indent=2)
print(f"replaced {len(tiled)} VAEDecodeTiled node(s) -> VAEDecode; wrote {dst}")

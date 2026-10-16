"""Turn the patched Workflow 02 into a max-quality variant (for a 96 GB GPU).

Usage: python3 tools/make_max_quality.py <patched.json> <max_quality.json> [shorter_size=1440]

  - UNET: int8 -> bf16 distilled transformer      (names taken from the model list in your Alpha Gen workflow)
  - Text encoder: int8 -> bf16 Gemma
  - VAEDecodeTiled -> VAEDecode (no tile seams, no temporal tiling)
  - Both Resize nodes -> shorter_size (must be a multiple of 32; less upscaling blur at the end)
  - Adds a 16-bit PNG sequence save of the final frames (lossless, for comp/grade)
"""
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
size = int(sys.argv[3]) if len(sys.argv) > 3 else 1440
assert size % 32 == 0, "shorter_size must be a multiple of 32"

d = json.load(open(src))
nodes = {n["id"]: n for n in d["nodes"]}
links = {l[0]: l for l in d["links"]}

nodes["5004_5602"]["widgets_values"][0] = "ltx-2.5-22b-distilled-transformer-bf16.safetensors"
nodes["5004_5602"]["widgets_values_named"]["unet_name"] = "ltx-2.5-22b-distilled-transformer-bf16.safetensors"
nodes["5004_5605"]["widgets_values"][0] = "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
nodes["5004_5605"]["widgets_values_named"]["clip_name"] = "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"

for rid in (5006, "5548_5026"):
    nodes[rid]["widgets_values"][1] = size
    nodes[rid]["widgets_values_named"]["resize_type.shorter_size"] = size

dec = nodes["5518_5538"]
dec["type"] = "VAEDecode"
dec["properties"]["Node name for S&R"] = "VAEDecode"
dec["widgets_values"], dec["widgets_values_named"] = [], {}
dec["size"] = [270, 46]
dec["title"] = "VAE Decode (full, no tiling)"

# lossless 16-bit PNG sequence of the final (restored, upscaled) frames
last = d["last_node_id"] + 1
scale = nodes[5011]
d["nodes"].append({
    "id": last, "type": "SaveImageAdvanced", "pos": [4100, 700], "size": [320, 300], "flags": {},
    "order": 200, "mode": 0, "title": "Save PNG sequence (16-bit)",
    "inputs": [{"name": "images", "type": "IMAGE", "link": None}],
    "outputs": [{"name": "images", "type": "IMAGE", "links": None}],
    "properties": {"Node name for S&R": "SaveImageAdvanced"},
    "widgets_values": ["LTX_Krish/frame", "png", "16-bit", "sRGB"],
})
lid = d["last_link_id"] + 1
links[lid] = [lid, 5011, 0, last, 0, "IMAGE"]
scale["outputs"][0]["links"].append(lid)
d["nodes"][-1]["inputs"][0]["link"] = lid
d["last_node_id"], d["last_link_id"] = last, lid

d["links"] = sorted(links.values(), key=lambda l: l[0])
by = {n["id"]: n for n in d["nodes"]}
for i, s, ss, t, ts, _ in d["links"]:
    assert i in by[s]["outputs"][ss]["links"], i
    assert by[t]["inputs"][ts]["link"] == i, i
json.dump(d, open(dst, "w"), indent=2)
print(f"wrote {dst} (shorter side {size})")

"""Turn the patched Workflow 02 into a max-quality variant (for a 96 GB GPU).

Usage: python3 tools/make_max_quality.py <patched.json> <max_quality.json> [shorter_size=1440] [dev|distilled]

  dev (default): ltx-2.5-22b-dev-transformer-bf16 with a real schedule (LTXVScheduler, 40 steps), CFG 4, euler.
  distilled:     keeps the distilled 8-step ManualSigmas / CFG 1 setup, only swapping to the bf16 distilled file.

  - UNET: int8 -> bf16 transformer (dev or distilled)
  - Text encoder: int8 -> bf16 Gemma
  - VAEDecodeTiled -> VAEDecode (no tile seams, no temporal tiling)
  - Both Resize nodes -> shorter_size (must be a multiple of 32; less upscaling blur at the end)
  - Adds a 16-bit PNG sequence save of the final frames (lossless, for comp/grade)
"""
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
size = int(sys.argv[3]) if len(sys.argv) > 3 else 1440
mode = sys.argv[4] if len(sys.argv) > 4 else "dev"
assert mode in ("dev", "distilled")
assert size % 32 == 0, "shorter_size must be a multiple of 32"

d = json.load(open(src))
nodes = {n["id"]: n for n in d["nodes"]}
links = {l[0]: l for l in d["links"]}

unet = ("ltx-2.5-22b-dev-transformer-bf16.safetensors" if mode == "dev"
        else "ltx-2.5-22b-distilled-transformer-bf16.safetensors")
nodes["5004_5602"]["widgets_values"][0] = unet
nodes["5004_5602"]["widgets_values_named"]["unet_name"] = unet
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

if mode == "dev":
    # real schedule instead of the distilled 8-step ManualSigmas; shift is sized from the video latent
    lid0 = d["last_link_id"] + 1
    d["last_link_id"] = lid0
    sch = nodes["5516_4984"]
    sch["type"] = "LTXVScheduler"
    sch["properties"]["Node name for S&R"] = "LTXVScheduler"
    sch["inputs"] = [{"name": "latent", "type": "LATENT", "shape": 7, "link": lid0}]
    sch["widgets_values"] = [40, 2.05, 0.95, True, 0.1]
    sch["widgets_values_named"] = {"steps": 40, "max_shift": 2.05, "base_shift": 0.95,
                                   "stretch": True, "terminal": 0.1}
    sch["size"] = [270, 150]
    links[lid0] = [lid0, "9002_3059", 0, "5516_4984", 0, "LATENT"]
    nodes["9002_3059"]["outputs"][0]["links"].append(lid0)
    nodes["5516_4828"]["widgets_values"] = [4]           # CFG
    nodes["5516_4828"]["widgets_values_named"] = {"cfg": 4}
    nodes["5516_4831"]["widgets_values"] = ["euler"]
    nodes["5516_4831"]["widgets_values_named"] = {"sampler_name": "euler"}
    neg = "worst quality, inconsistent motion, blurry, jittery, distorted, watermark, flicker"
    nodes["5014_2612"]["widgets_values"] = [neg]
    nodes["5014_2612"]["widgets_values_named"] = {"text": neg}

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

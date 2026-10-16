"""Patch 'LTX Workflow 02' (flat graph, video + frozen source audio) for exact frame count/size/audio.

Usage: python3 tools/patch_workflow_02.py <original.json> <patched.json>

  1. Krish LTX Frame Prep after the 1088 resize -> latent length becomes a valid 8n+1, size a multiple of 32.
  2. Krish LTX Audio Prep pads the source audio to the padded video length (audio/video latents match).
  3. Krish LTX Frame Restore + Parity Check after VAE decode -> exactly the original frames and size.
  4. Krish LTX Audio Restore trims the output audio to the original duration.
  5. Final upscale lanczos (was nearest-exact); SaveVideo gets a filename prefix (was empty).
"""
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
d = json.load(open(src))
nodes = {n["id"]: n for n in d["nodes"]}
links = {l[0]: l for l in d["links"]}
next_node, next_link = [d["last_node_id"]], [d["last_link_id"]]
KRISH = "#2a4a3a", "#1f3a2c"


def new_node(type_, pos, size, inputs, outputs, widgets, title, color=KRISH):
    next_node[0] += 1
    n = {"id": next_node[0], "type": type_, "pos": pos, "size": size, "flags": {},
         "order": 100 + next_node[0] - d["last_node_id"], "mode": 0, "inputs": inputs,
         "outputs": [{"name": a, "type": b, "links": []} for a, b in outputs],
         "title": title, "properties": {"Node name for S&R": type_}, "widgets_values": widgets}
    if color:
        n["color"], n["bgcolor"] = color
    d["nodes"].append(n)
    nodes[n["id"]] = n
    return n


def inp(name, type_, widget=False, optional=False):
    i = {"name": name, "type": type_, "link": None}
    if widget:
        i["widget"] = {"name": name}
    if optional:
        i["shape"] = 7
    return i


def out_links(node, slot):
    o = node["outputs"][slot]
    if o.get("links") is None:
        o["links"] = []
    return o["links"]


def connect(a, a_slot, b, b_name):
    next_link[0] += 1
    lid = next_link[0]
    b_slot = next(i for i, s in enumerate(b["inputs"]) if s["name"] == b_name)
    links[lid] = [lid, a["id"], a_slot, b["id"], b_slot, a["outputs"][a_slot]["type"]]
    out_links(a, a_slot).append(lid)
    b["inputs"][b_slot]["link"] = lid


def reroute_origin(lid, new_src, slot):
    l = links[lid]
    out_links(nodes[l[1]], l[2]).remove(lid)
    l[1], l[2] = new_src["id"], slot
    out_links(new_src, slot).append(lid)


resize2 = nodes["5548_5026"]       # 2nd resize (1088) -> guide image (28) + GetImageSize (38)
vhs, vinfo = nodes[5002], nodes[5003]
vae_dec = nodes["5518_5538"]       # VAEDecodeTiled -> ImageScale (62)
aud_dec = nodes["5518_4848"]       # LTXVAudioVAEDecode -> CreateVideo (45)

prep = new_node("KrishLTXFramePrep", [700, 130], [320, 190], [inp("images", "IMAGE")],
                [("images", "IMAGE"), ("conform_info", "KRISH_CONFORM"), ("num_frames", "INT"),
                 ("width", "INT"), ("height", "INT")],
                [8, 32, "repeat_last", "reflect", "center"], "Krish LTX Frame Prep")
aprep = new_node("KrishLTXAudioPrep", [1100, 1005], [270, 100],
                 [inp("audio", "AUDIO"), inp("conform_info", "KRISH_CONFORM"),
                  inp("fps", "FLOAT")], [("audio", "AUDIO")], [], "Krish LTX Audio Prep")
restore = new_node("KrishLTXFrameRestore", [3756, 330], [320, 190],
                   [inp("generated", "IMAGE"), inp("conform_info", "KRISH_CONFORM"),
                    inp("source_images", "IMAGE", optional=True), inp("mask", "MASK", optional=True)],
                   [("images", "IMAGE"), ("report", "STRING")], ["nearest", "bilinear"],
                   "Krish LTX Frame Restore")
parity = new_node("KrishFrameParityCheck", [3756, 560], [300, 130],
                  [inp("source", "IMAGE"), inp("result", "IMAGE")],
                  [("result", "IMAGE"), ("report", "STRING")], ["error"], "Krish Frame Parity Check")
arest = new_node("KrishLTXAudioRestore", [3750, 420 + 330], [270, 100],
                 [inp("audio", "AUDIO"), inp("conform_info", "KRISH_CONFORM"),
                  inp("fps", "FLOAT")], [("audio", "AUDIO")], [], "Krish LTX Audio Restore")
arest["pos"] = [3453, 420]
rep1 = new_node("PreviewAny", [4090, 330], [300, 90], [inp("source", "*")], [], [], "Restore report", None)
rep2 = new_node("PreviewAny", [4090, 440], [300, 90], [inp("source", "*")], [], [], "Parity report", None)
new_node("Note", [700, -150], [420, 230], [], [],
         ["KRISH FIXES\n"
          "- Prep pads frames to 8n+1 and size to a multiple of 32 BEFORE the latent is sized (nothing dropped).\n"
          "- Audio Prep pads the source audio to the padded video length.\n"
          "- Restore trims frames back to the exact original count/size; Parity errors if they differ.\n"
          "- Audio Restore trims the audio to the original duration.\n"
          "- Final upscale: lanczos. Keep both Resize nodes at the same value (a multiple of 32)."],
         "READ ME", ("#432", "#653"))

# 1. Prep after resize2: guide image + latent size now read the padded frames
connect(resize2, 0, prep, "images")
reroute_origin(28, prep, 0)
reroute_origin(38, prep, 0)

# 2. audio prep
connect(vhs, 2, aprep, "audio")
connect(prep, 1, aprep, "conform_info")
connect(vinfo, 0, aprep, "fps")
reroute_origin(57, aprep, 0)

# 3. restore + parity before final upscale
connect(vae_dec, 0, restore, "generated")
connect(prep, 1, restore, "conform_info")
connect(restore, 0, parity, "result")
connect(resize2, 0, parity, "source")
reroute_origin(62, parity, 0)
connect(restore, 1, rep1, "source")
connect(parity, 1, rep2, "source")

# 4. audio restore
connect(aud_dec, 0, arest, "audio")
connect(prep, 1, arest, "conform_info")
connect(vinfo, 0, arest, "fps")
reroute_origin(45, arest, 0)

# 5. small fixes
nodes[5011]["widgets_values"][0] = "lanczos"
nodes[5011]["widgets_values_named"]["upscale_method"] = "lanczos"
nodes[4852]["widgets_values"][0] = "video/LTX_Krish"
nodes[4852]["widgets_values_named"]["filename_prefix"] = "video/LTX_Krish"

d["links"] = sorted(links.values(), key=lambda l: l[0])
d["last_node_id"], d["last_link_id"] = next_node[0], next_link[0]
for lid, s, ss, t, ts, _ in d["links"]:
    assert lid in nodes[s]["outputs"][ss]["links"], lid
    assert nodes[t]["inputs"][ts]["link"] == lid, lid
json.dump(d, open(dst, "w"), indent=2)
print(f"wrote {dst}: {len(d['nodes'])} nodes, {len(d['links'])} links")

"""Patch the 'LTX 2.5 Alpha Gen' workflow so output frame count/size/audio match the input exactly.

Usage: python3 tools/patch_alpha_gen_workflow.py <original.json> <patched.json>

Changes (everything else, incl. subgraphs, models and LoRA, is left untouched):
  1. VHS Load Video: frame_load_cap 2 -> 0 (load every frame).
  2. Krish LTX Frame Prep inserted between the resize and the Source Video subgraph, so the
     generation length is a valid 8n+1 and the size a multiple of 32 (nothing is cropped).
  3. Krish LTX Frame Restore + Parity Check after VAE Decode, feeding every old consumer.
  4. Krish LTX Audio Restore trims the generated audio to the original duration.
  5. Final upscale uses lanczos instead of nearest-exact.
"""
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
d = json.load(open(src))
nodes = {n["id"]: n for n in d["nodes"]}
links = {l[0]: l for l in d["links"]}

next_node = [d["last_node_id"]]
next_link = [d["last_link_id"]]
KRISH = "#2a4a3a", "#1f3a2c"


def new_node(type_, pos, size, inputs, outputs, widgets, title, color=KRISH):
    next_node[0] += 1
    n = {"id": next_node[0], "type": type_, "pos": pos, "size": size, "flags": {},
         "order": 100 + next_node[0] - d["last_node_id"], "mode": 0,
         "inputs": inputs, "outputs": [{"name": a, "type": b, "links": []} for a, b in outputs],
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


def connect(a, a_slot, b, b_name):
    """New link a.out[a_slot] -> b.input b_name."""
    next_link[0] += 1
    lid = next_link[0]
    b_slot = next(i for i, s in enumerate(b["inputs"]) if s["name"] == b_name)
    typ = a["outputs"][a_slot]["type"]
    links[lid] = [lid, a["id"], a_slot, b["id"], b_slot, typ]
    a["outputs"][a_slot].setdefault("links", []).append(lid)
    b["inputs"][b_slot]["link"] = lid
    return lid


def reroute_origin(lid, new_src, slot):
    """Keep link id (and its target) but change where it comes from."""
    l = links[lid]
    old = nodes[l[1]]
    old["outputs"][l[2]]["links"].remove(lid)
    l[1], l[2] = new_src["id"], slot
    new_src["outputs"][slot].setdefault("links", []).append(lid)


def drop_link(lid):
    l = links.pop(lid)
    nodes[l[1]]["outputs"][l[2]]["links"].remove(lid)
    nodes[l[3]]["inputs"][l[4]]["link"] = None


# ---- 1. load every frame
load = nodes[9095]
load["widgets_values"]["frame_load_cap"] = 0
load["widgets_values"]["videopreview"]["params"]["frame_load_cap"] = 0
if "widgets_values_named" in load:
    load["widgets_values_named"]["frame_load_cap"] = 0
    load["widgets_values_named"]["videopreview"]["params"]["frame_load_cap"] = 0

resize, create_in, vae_dec = nodes[9052], nodes[9010], nodes[9107]
srcvid = nodes[5548]  # Source Video subgraph instance (out 4 = fps)
audio_dec = nodes[9105]

# ---- new nodes
prep = new_node("KrishLTXFramePrep", [-4300, 3560], [320, 190],
                [inp("images", "IMAGE")],
                [("images", "IMAGE"), ("conform_info", "KRISH_CONFORM"), ("num_frames", "INT"),
                 ("width", "INT"), ("height", "INT")],
                [8, 32, "repeat_last", "reflect", "center"], "Krish LTX Frame Prep")
restore = new_node("KrishLTXFrameRestore", [-1000, 3190], [320, 190],
                   [inp("generated", "IMAGE"), inp("conform_info", "KRISH_CONFORM"),
                    inp("source_images", "IMAGE", optional=True), inp("mask", "MASK", optional=True)],
                   [("images", "IMAGE"), ("report", "STRING")],
                   ["nearest", "bilinear"], "Krish LTX Frame Restore")
parity = new_node("KrishFrameParityCheck", [-1000, 3420], [300, 130],
                  [inp("source", "IMAGE"), inp("result", "IMAGE")],
                  [("result", "IMAGE"), ("report", "STRING")], ["error"], "Krish Frame Parity Check")
arest = new_node("KrishLTXAudioRestore", [-1000, 2900], [300, 110],
                 [inp("audio", "AUDIO"), inp("conform_info", "KRISH_CONFORM"),
                  inp("fps", "FLOAT")],
                 [("audio", "AUDIO")], [], "Krish LTX Audio Restore")
rep1 = new_node("PreviewAny", [-1000, 3580], [300, 90], [inp("source", "*")], [], [], "Restore report", None)
rep2 = new_node("PreviewAny", [-1000, 3690], [300, 90], [inp("source", "*")], [], [], "Parity report", None)
note = new_node("Note", [-4300, 3300], [420, 230], [], [],
                ["KRISH FIXES\n"
                 "- Load Video: frame_load_cap was 2 -> now 0 (all frames).\n"
                 "- Prep pads frames to 8n+1 and size to a multiple of 32 (no cropping/dropping).\n"
                 "- Restore trims back to the exact original frame count and size; Parity Check errors if not.\n"
                 "- Audio Restore trims audio to the original duration.\n"
                 "- Keep the 'shorter size limit' (1088) equal to the Resize node and a multiple of 32."],
                "READ ME", ("#432", "#653"))

# ---- 2. Prep between resize and Source Video
drop_link(13857)  # resize -> CreateVideo.images
connect(resize, 0, prep, "images")
connect(prep, 0, create_in, "images")

# ---- 3. Restore + Parity after VAE decode; old consumers now read from Parity
for lid in (13929, 13930, 13934):
    reroute_origin(lid, parity, 0)
connect(vae_dec, 0, restore, "generated")
connect(prep, 1, restore, "conform_info")
connect(restore, 0, parity, "result")
connect(resize, 0, parity, "source")
connect(restore, 1, rep1, "source")
connect(parity, 1, rep2, "source")

# ---- 4. audio
for lid in (13922, 13938):
    reroute_origin(lid, arest, 0)
connect(audio_dec, 0, arest, "audio")
connect(prep, 1, arest, "conform_info")
connect(srcvid, 4, arest, "fps")

# ---- 5. final upscale: lanczos
nodes[9109]["widgets_values"][0] = "lanczos"

d["links"] = sorted(links.values(), key=lambda l: l[0])
d["last_node_id"], d["last_link_id"] = next_node[0], next_link[0]

# ---- self-check
for l in d["links"]:
    lid, s, ss, t, ts, _ = l
    assert lid in nodes[s]["outputs"][ss]["links"], l
    assert nodes[t]["inputs"][ts]["link"] == lid, l
json.dump(d, open(dst, "w"), indent=2)
print(f"wrote {dst}: {len(d['nodes'])} nodes, {len(d['links'])} links")

"""Patch 'LTX 2.5 Matte Gen - No audio' for VFX: no audio in/out and strong frame-to-frame QC.

Usage: python3 tools/patch_matte_no_audio.py <original.json> <patched.json>

  1. Audio removed from the deliverable: LTXVAudioVAEDecode, its links and the stray LoadAudio node are deleted;
     CreateVideo gets no audio. (An EMPTY audio latent is still built inside the Preprocess subgraph because the
     LTX-2 model is a joint audio-video model; it is never decoded, saved or exported.)
  2. Krish Matte Frame-to-Frame Check after the existing Parity Check (source = resized original):
     count, NaN/Inf, range, time shift (lag test), duplicated frames, isolated dropouts, flicker spikes.
  3. Second Krish Parity Check at the very end: final output vs the ORIGINAL loaded frames
     (same frame count, width, height, channels as the clip you gave it).
  4. 16-bit PNG matte sequence saved from the verified final frames; SaveVideo prefix set.
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


def inp(name, type_):
    return {"name": name, "type": type_, "link": None}


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


def drop_link(lid):
    l = links.pop(lid)
    out_links(nodes[l[1]], l[2]).remove(lid)
    nodes[l[3]]["inputs"][l[4]]["link"] = None


vhs, resize = nodes[9095], nodes[9052]
parity1, scale, mkvideo, savevid = nodes[9119], nodes[9109], nodes[9111], nodes[9112]

# ---- 1. remove audio from the deliverable
adec = nodes[9105]
for lid in [l[0] for l in list(links.values()) if adec["id"] in (l[1], l[3])]:
    drop_link(lid)
d["nodes"] = [n for n in d["nodes"] if n["id"] not in (9105, 9126)]  # audio decode + stray LoadAudio
for gone in (9105, 9126):
    nodes.pop(gone)
mkvideo["inputs"][1]["link"] = None  # CreateVideo.audio stays empty

# ---- 2/3. QC nodes
px, py = parity1["pos"]
check = new_node("KrishMatteFrameCheck", [px, py + 200], [320, 130],
                 [inp("source", "IMAGE"), inp("result", "IMAGE")],
                 [("result", "IMAGE"), ("report", "STRING"), ("flagged_frames", "STRING")],
                 ["error", 4], "Krish Matte Frame-to-Frame Check")
final = new_node("KrishFrameParityCheck", [scale["pos"][0] + 320, scale["pos"][1]], [300, 130],
                 [inp("source", "IMAGE"), inp("result", "IMAGE")],
                 [("result", "IMAGE"), ("report", "STRING")], ["error"], "Krish FINAL Parity (vs original)")
rep_a = new_node("PreviewAny", [px + 340, py + 200], [320, 140], [inp("source", "*")], [], [], "Matte check report", None)
rep_b = new_node("PreviewAny", [scale["pos"][0] + 640, scale["pos"][1]], [300, 90],
                 [inp("source", "*")], [], [], "Final parity report", None)
save_seq = new_node("SaveImageAdvanced", [mkvideo["pos"][0], mkvideo["pos"][1] + 220], [320, 300],
                    [inp("images", "IMAGE")], [("images", "IMAGE")],
                    ["LTX_Matte/matte", "png", "16-bit", "sRGB"], "Save matte PNG sequence (16-bit)", None)

# parity1 -> matte check -> ImageScale
connect(resize, 0, check, "source")
reroute_origin(13934, check, 0)           # ImageScale.image now reads the checked frames
connect(parity1, 0, check, "result")
connect(check, 1, rep_a, "source")
# ImageScale -> FINAL parity (vs original loaded frames) -> CreateVideo + PNG sequence
connect(scale, 0, final, "result")
connect(vhs, 0, final, "source")          # original frames at original resolution
connect(final, 1, rep_b, "source")
reroute_origin(13940, final, 0)           # CreateVideo.images
connect(final, 0, save_seq, "images")

# ---- 4. small fixes
savevid["widgets_values"][0] = "video/LTX_Matte"
savevid["widgets_values_named"]["filename_prefix"] = "video/LTX_Matte"
nodes[9123]["widgets_values"] = [
    "KRISH MATTE GEN - NO AUDIO (VFX)\n"
    "- No audio in or out: audio decode and LoadAudio removed, CreateVideo has no audio.\n"
    "  (The model still builds an EMPTY audio latent internally; it is never decoded or saved.)\n"
    "- Prep pads frames to 8n+1 / size to x32; Restore trims back to the exact original frames.\n"
    "- Parity Check #1 + Matte Frame-to-Frame Check: count, NaN/range, time shift, duplicates, dropouts, flicker.\n"
    "- FINAL Parity compares the deliverable with the ORIGINAL loaded frames (count, W, H, channels).\n"
    "- Any failed check stops the run with a clear error. 16-bit PNG matte sequence is saved too.\n"
    "- Keep 'shorter size limit' equal to the Resize node (multiple of 32)."]

d["links"] = sorted(links.values(), key=lambda l: l[0])
d["last_node_id"], d["last_link_id"] = next_node[0], next_link[0]
for lid, s, ss, t, ts, _ in d["links"]:
    assert lid in nodes[s]["outputs"][ss]["links"], lid
    assert nodes[t]["inputs"][ts]["link"] == lid, lid
json.dump(d, open(dst, "w"), indent=2)
print(f"wrote {dst}: {len(d['nodes'])} nodes, {len(d['links'])} links")

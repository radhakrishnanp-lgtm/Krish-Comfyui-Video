"""Patch the DepthCrafter workflow: guarantee output frames/size == input, show fps. Nothing else is touched.

Usage: python3 tools/patch_depthcrafter.py <original.json> <patched.json>

DepthCrafter (akatz-ai/ComfyUI-DepthCrafter-Nodes) already returns exactly one depth map per input frame
(sliding-window stitching decodes `num_frames`) and the workflow scales back to the loaded size, so no padding
(Prep/Restore) is needed. This adds a FINAL Krish Parity Check: the finished frames are compared with the
ORIGINAL loaded frames (count, width, height, channels) and the run stops with a clear error if they differ.
The Create Video fps already comes from the source video; two displays show fps and frame count.
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


vhs, info, rtx, mkvideo = nodes[3], nodes[23], nodes[29], nodes[27]
x, y = mkvideo["pos"]

parity = new_node("KrishFrameParityCheck", [x - 20, y + 360], [300, 130],
                  [{"name": "source", "type": "IMAGE", "link": None},
                   {"name": "result", "type": "IMAGE", "link": None}],
                  [("result", "IMAGE"), ("report", "STRING")], ["error"], "Krish FINAL Parity (vs original)")
rep = new_node("PreviewAny", [x + 300, y + 360], [320, 100], [{"name": "source", "type": "*", "link": None}],
               [], [], "Frames / size check", None)
fps_view = new_node("PreviewAny", [x + 300, y + 480], [320, 90], [{"name": "source", "type": "*", "link": None}],
                    [], [], "Output fps (= source fps)", None)
cnt_view = new_node("PreviewAny", [x + 300, y + 590], [320, 90], [{"name": "source", "type": "*", "link": None}],
                    [], [], "Frames loaded from the video", None)
new_node("Note", [-748, 40], [420, 150], [], [],
         ["KRISH EXACT-OUTPUT CHECK\n"
          "DepthCrafter returns one depth map per input frame. The FINAL Parity check compares the finished "
          "frames with the ORIGINAL loaded video (frame count, width, height) and stops with an error if they "
          "differ. Output fps is the source fps. Keep Load Video: frame_load_cap 0, select_every_nth 1, "
          "force_rate 0, otherwise the loaded frames are not the full clip."],
         "READ ME", ("#432", "#653"))

connect(vhs, 0, parity, "source")          # original frames at original resolution
connect(rtx, 0, parity, "result")
reroute_origin(19, parity, 0)              # CreateVideo.images now reads the verified frames
connect(parity, 1, rep, "source")
connect(info, 0, fps_view, "source")       # source fps (the value Create Video uses)
connect(info, 6, cnt_view, "source")       # loaded_frame_count

d["links"] = sorted(links.values(), key=lambda l: l[0])
d["last_node_id"], d["last_link_id"] = next_node[0], next_link[0]
for lid, s, ss, t, ts, _ in d["links"]:
    assert lid in nodes[s]["outputs"][ss]["links"], lid
    assert nodes[t]["inputs"][ts]["link"] == lid, lid
json.dump(d, open(dst, "w"), indent=2)
print(f"wrote {dst}: {len(d['nodes'])} nodes, {len(d['links'])} links")

"""Build an EXR-in / EXR-out copy of the 'Qwen 2.1 - edit' inpaint workflow. The original file is not changed.

Usage: python3 tools/make_qwen_exr_workflow.py <qwen_edit.json> <out.json> [exr_name]

  EXR plate (scene-linear, any value range) -> Krish EXR Linear->Display (sRGB + soft knee, nothing clipped)
  -> InpaintCrop -> Qwen edit -> InpaintStitch -> Krish EXR Keep Original Outside Edit (decodes only what Qwen changed;
  every other pixel is the original linear pixel, bit for bit) -> Krish EXR Save (same colour tags, metadata, channel
  type, compression, frame number as the source; header verified).
  Removed: the 8-bit PNG plate loader (node 500). Kept as they were: the mask loader, every Qwen/Ollama/crop/stitch node
  and the PNG save (now a display-encoded 8-bit preview).
"""
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
exr_name = sys.argv[3] if len(sys.argv) > 3 else "TD_005_290_MP001.1001.exr"
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


def inp(name, type_, optional=False):
    i = {"name": name, "type": type_, "link": None}
    if optional:
        i["shape"] = 7
    return i


def connect(a, a_slot, b, b_name):
    next_link[0] += 1
    lid = next_link[0]
    b_slot = next(i for i, s in enumerate(b["inputs"]) if s["name"] == b_name)
    links[lid] = [lid, a["id"], a_slot, b["id"], b_slot, a["outputs"][a_slot]["type"]]
    if a["outputs"][a_slot].get("links") is None:
        a["outputs"][a_slot]["links"] = []
    a["outputs"][a_slot]["links"].append(lid)
    b["inputs"][b_slot]["link"] = lid


crop, stitch, png_save = nodes[493], nodes[502], nodes[461]

load = new_node("KrishEXRLoad", [-1100, 100], [360, 300], [],
                [("images", "IMAGE"), ("alpha", "MASK"), ("exr_meta", "EXR_META"), ("info", "STRING"),
                 ("frame_count", "INT"), ("width", "INT"), ("height", "INT"), ("fps", "FLOAT")],
                [exr_name, False, -1, -1, 1, "", 0, 24.0], "Krish EXR Load (plate)")
enc = new_node("KrishEXRToDisplay", [-700, 100], [300, 150], [inp("images", "IMAGE")], [("images", "IMAGE")],
               ["sRGB", 0.0, "soft_knee"], "Krish EXR Linear -> Display (model input)")
merge = new_node("KrishEXRKeepOriginalOutsideEdit", [3700, 330], [360, 250],
                 [inp("original_linear", "IMAGE"), inp("original_encoded", "IMAGE"), inp("edited_encoded", "IMAGE")],
                 [("images", "IMAGE"), ("changed_mask", "MASK"), ("report", "STRING")],
                 ["sRGB", 0.0, "soft_knee", 1e-06], "Krish EXR Keep Original Outside Edit")
save = new_node("KrishEXRSave", [4100, 330], [360, 400],
                [inp("images", "IMAGE"), inp("exr_meta", "EXR_META"), inp("matte", "MASK", optional=True)],
                [("paths", "STRING"), ("report", "STRING")],
                ["replace_rgb", "krish_exr", "_qwen", "same_as_source", "same_as_source", "error", "overwrite", "strict",
                 True, ""], "Krish EXR Save (same details as source)")
info_view = new_node("PreviewAny", [-1100, 440], [440, 200], [inp("source", "*")], [], [], "EXR info (colour hint, value range)", None)
merge_view = new_node("PreviewAny", [3700, 620], [360, 110], [inp("source", "*")], [], [], "Edit coverage", None)
save_view = new_node("PreviewAny", [4100, 760], [360, 170], [inp("source", "*")], [], [], "EXR save report", None)
new_node("Note", [-1100, -230], [560, 290], [], [],
         ["EXR IN / EXR OUT (Qwen 2.1 edit)\n"
          "0. Needs Krish-Comfyui-Video 0.4.0+. At start-up the console prints '[Krish] ... merge node: yes'. A red Krish node with UNKNOWN widgets means ComfyUI is still running an older copy: update the pack and restart.\n"
          "1. Krish EXR Load: click 'choose EXR file(s) to upload' at the bottom of the node (or type a path). Single frame; the mask below must be the same size as the EXR.\n"
          "2. Linear -> Display (sRGB + soft_knee) gives Qwen a normal-looking image without clipping highlights or negatives.\n"
          "3. After the stitch, 'Keep Original Outside Edit' decodes ONLY the pixels Qwen changed; all other pixels are the original linear pixels, bit for bit.\n"
          "4. Krish EXR Save writes the same frame number, colour tags, metadata, half/float type and compression as the source ('_qwen' suffix) and verifies the header.\n"
          "The 8-bit PNG save is now only a display-encoded preview."],
         "READ ME", ("#432", "#653"))

# the crop now reads the encoded EXR instead of the PNG plate
l22 = links[22]
nodes[500]["outputs"][0]["links"].remove(22)
l22[1], l22[2] = enc["id"], 0
enc["outputs"][0]["links"].append(22)
d["nodes"] = [n for n in d["nodes"] if n["id"] != 500]
nodes.pop(500)

connect(load, 0, enc, "images")
connect(load, 0, merge, "original_linear")
connect(enc, 0, merge, "original_encoded")
connect(stitch, 0, merge, "edited_encoded")
connect(merge, 0, save, "images")
connect(load, 2, save, "exr_meta")
connect(load, 3, info_view, "source")
connect(merge, 2, merge_view, "source")
connect(save, 1, save_view, "source")
png_save["title"] = "Preview PNG (display-encoded, 8-bit)"

d["links"] = sorted(links.values(), key=lambda l: l[0])
d["last_node_id"], d["last_link_id"] = next_node[0], next_link[0]
for lid, s, ss, t, ts, _ in d["links"]:
    assert lid in nodes[s]["outputs"][ss]["links"], lid
    assert nodes[t]["inputs"][ts]["link"] == lid, lid
json.dump(d, open(dst, "w"), indent=2)
print(f"wrote {dst}: {len(d['nodes'])} nodes, {len(d['links'])} links")

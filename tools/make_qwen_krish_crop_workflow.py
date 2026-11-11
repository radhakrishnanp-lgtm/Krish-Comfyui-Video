"""Swap the inpaint crop/stitch nodes of the Qwen 2.1 EXR workflow for Krish Inpaint Crop / Stitch.

Usage: python3 tools/make_qwen_krish_crop_workflow.py <Qwen_2.1_Edit_EXR.json> <out.json>

  InpaintCropImproved  -> Krish Inpaint Crop   (same first three outputs: stitcher, cropped_image, cropped_mask)
  InpaintStitchImproved-> Krish Inpaint Stitch (same inputs: stitcher, inpainted_image)
  ImageScaleToTotalPixels (nearest-exact) is removed: the crop already delivers an exact ~2 MP image whose size is a
  multiple of 16, so the Qwen latent size equals the crop size and nothing is rescaled between the crop and the model.
Everything else (EXR load/encode, Qwen, Ollama, merge, EXR save) is unchanged.
"""
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
d = json.load(open(src))
nodes = {n["id"]: n for n in d["nodes"]}
links = {l[0]: l for l in d["links"]}
next_node, next_link = [d["last_node_id"]], [d["last_link_id"]]
KRISH = "#2a4a3a", "#1f3a2c"

crop, stitch, rescale, getsize = nodes[493], nodes[502], nodes[487], nodes[477]
assert crop["type"] == "InpaintCropImproved" and stitch["type"] == "InpaintStitchImproved"


def retype(n, type_, title, inputs, outputs, widgets, size):
    n["type"], n["title"], n["widgets_values"] = type_, title, widgets
    n.pop("widgets_values_named", None)
    n["properties"] = {"Node name for S&R": type_}
    n["inputs"], n["outputs"] = inputs, outputs
    n["size"], n["color"], n["bgcolor"] = size, *KRISH


def out(name, type_, links_=None):
    return {"name": name, "type": type_, "links": links_}


# ---- crop (inputs image/mask keep their links; the unused optional_context_mask slot is dropped)
c_in = [dict(i) for i in crop["inputs"][:2]]
c_old = {o["name"]: o.get("links") for o in crop["outputs"]}
retype(crop, "KrishInpaintCrop", "Krish Inpaint Crop", c_in,
       [out("stitcher", "KRISH_STITCHER", c_old["stitcher"]), out("cropped_image", "IMAGE", c_old["cropped_image"]),
        out("cropped_mask", "MASK", c_old["cropped_mask"]), out("preview", "IMAGE", []), out("info", "STRING", [])],
       [0.01, True, 0, 1.3, 32, "megapixels", 2.0, 1536, 1024, 1024, 16, 4.0, "reflect", "bicubic", "binary (1 MP = 1024x1024)", "same as edit region"], [380, 520])

# ---- stitch (stitcher / inpainted_image keep their links; image output keeps its links)
s_old = {o["name"]: o.get("links") for o in stitch["outputs"]}
retype(stitch, "KrishInpaintStitch", "Krish Inpaint Stitch", [dict(i) for i in stitch["inputs"]],
       [out("image", "IMAGE", s_old["image"]), out("blend_mask", "MASK", []), out("report", "STRING", [])],
       ["mask_feather", "auto", 24, "affine", 64, "low_frequency", 1.0, "bicubic", True], [330, 300])

# ---- the rescale node goes; the latent size now comes straight from the exact crop
del links[16]
crop["outputs"][1]["links"].remove(16)
del nodes[487]
d["nodes"] = [n for n in d["nodes"] if n["id"] != 487]
links[2][1], links[2][2] = crop["id"], 1
crop["outputs"][1]["links"].append(2)
links[26][5] = "KRISH_STITCHER"


def new_node(type_, pos, size, inputs, widgets, title, color=None):
    next_node[0] += 1
    n = {"id": next_node[0], "type": type_, "pos": pos, "size": size, "flags": {}, "order": 100 + next_node[0] - d["last_node_id"],
         "mode": 0, "inputs": inputs, "outputs": [], "title": title, "properties": {"Node name for S&R": type_},
         "widgets_values": widgets}
    if color:
        n["color"], n["bgcolor"] = color
    d["nodes"].append(n)
    nodes[n["id"]] = n
    return n


def connect(a, slot, b, name):
    next_link[0] += 1
    lid = next_link[0]
    b_slot = next(i for i, s in enumerate(b["inputs"]) if s["name"] == name)
    links[lid] = [lid, a["id"], slot, b["id"], b_slot, a["outputs"][slot]["type"]]
    a["outputs"][slot]["links"] = (a["outputs"][slot].get("links") or []) + [lid]
    b["inputs"][b_slot]["link"] = lid


info_view = new_node("PreviewAny", [1232, 670], [380, 110], [{"name": "source", "type": "*", "link": None}], [], "Crop info")
pv = new_node("PreviewImage", [1232, 800], [380, 280], [{"name": "images", "type": "IMAGE", "link": None}], [],
              "Crop preview (mask red, crop box green)")
rep_view = new_node("PreviewAny", [3670, 450], [330, 130], [{"name": "source", "type": "*", "link": None}], [], "Stitch report")
connect(crop, 4, info_view, "source")
connect(crop, 3, pv, "images")
connect(stitch, 2, rep_view, "source")

# the taller stitch node would overlap the EXR merge/save column: move that column down
for n in d["nodes"]:
    if n["type"] in ("KrishEXRKeepOriginalOutsideEdit", "KrishEXRSave") or (n["type"] == "PreviewAny" and n["pos"][0] >= 3690 and n["pos"][1] >= 600):
        n["pos"][1] += 330
rep_view["pos"][1] = 450

new_node("Note", [1232, -260], [640, 330], [],
         ["KRISH INPAINT CROP / STITCH (replaces InpaintCropImproved / InpaintStitchImproved + the nearest-exact rescale)\n"
          "- Crop: exact integer crop box around the mask (x context_scale), scaled to ~2 MP with a size that is a multiple of 16, so the Qwen latent size == the crop size and nothing is rescaled in between. Shrinking is antialiased, enlarging is bicubic, nothing is clipped.\n"
          "- Stitch: pixels outside the blend are the ORIGINAL pixels (bit for bit). The blend follows the mask shape (feather 'auto' = 10% of the mask size, exactly 0 beyond it).\n"
          "- Qwen re-renders the whole crop, so it drifts a little: the stitch measures a sub-pixel shift and a smooth colour/brightness difference on the UNTOUCHED surroundings and corrects them before blending (align / color_match; the Stitch report shows what was applied).\n"
          "- For pixel-exact work without any resize set size_mode to 'native (no resize)'.\n"
          "- Needs Krish-Comfyui-Video 0.6.4+ (console: '... inpaint nodes: yes')."],
         "READ ME (inpaint)", ("#432", "#653"))

d["links"] = sorted(links.values(), key=lambda l: l[0])
d["last_node_id"], d["last_link_id"] = next_node[0], next_link[0]
for lid, s, ss, t, ts, _ in d["links"]:
    assert lid in nodes[s]["outputs"][ss]["links"], lid
    assert nodes[t]["inputs"][ts]["link"] == lid, lid
json.dump(d, open(dst, "w"), indent=2)
print(f"wrote {dst}: {len(d['nodes'])} nodes, {len(d['links'])} links")

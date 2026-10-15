"""Generates workflows/krish_ltx_exact_frames_v2v.json (ComfyUI UI-format graph).

Run: python3 tools/build_workflow.py
"""
import json
import pathlib

nodes, links = [], []
_nid = [0]
_lid = [0]


def node(type_, pos, size, inputs, outputs, widgets=None, title=None, color=None):
    """inputs: [(name, type)] linked slots only; outputs: [(name, type)]."""
    _nid[0] += 1
    n = {
        "id": _nid[0], "type": type_, "pos": list(pos), "size": list(size),
        "flags": {}, "order": _nid[0], "mode": 0,
        "inputs": [{"name": a, "type": b, "link": None} for a, b in inputs],
        "outputs": [{"name": a, "type": b, "links": [], "slot_index": i}
                    for i, (a, b) in enumerate(outputs)],
        "properties": {"Node name for S&R": type_},
        "widgets_values": widgets or [],
    }
    if title:
        n["title"] = title
    if color:
        n["color"], n["bgcolor"] = color
    nodes.append(n)
    return n


def link(src, src_slot, dst, dst_name):
    _lid[0] += 1
    dst_slot = next(i for i, s in enumerate(dst["inputs"]) if s["name"] == dst_name)
    typ = src["outputs"][src_slot]["type"]
    links.append([_lid[0], src["id"], src_slot, dst["id"], dst_slot, typ])
    src["outputs"][src_slot]["links"].append(_lid[0])
    dst["inputs"][dst_slot]["link"] = _lid[0]


KRISH = ("#2a4a3a", "#1f3a2c")
IN, OUT = ("#2a3a5a", "#1f2c46"), ("#5a3a2a", "#46301f")

# ---- notes
node("Note", (40, -330), (560, 250), [], [], [
    "KRISH LTX EXACT-FRAMES V2V\n\n"
    "1. Pick your LTX checkpoint in 'Load Checkpoint' and the text encoder in 'Load CLIP' (type: ltxv).\n"
    "2. Load your video in 'Load Video'.\n"
    "3. Edit the prompt. Lower 'denoise' (0.3-0.6) keeps more of the original.\n"
    "4. Run. 'Frame Parity Check' errors if output frame count/size differs from the input.\n\n"
    "Green nodes = Krish Frame Conform pack (custom_nodes/krish_frame_conform)."], title="README")

# ---- load
ckpt = node("CheckpointLoaderSimple", (40, 0), (320, 100), [], [("MODEL", "MODEL"), ("CLIP", "CLIP"), ("VAE", "VAE")],
            ["ltx-video-2b-v0.9.5.safetensors"], title="Load Checkpoint (set your LTX)", color=IN)
clip = node("CLIPLoader", (40, 140), (320, 110), [], [("CLIP", "CLIP")],
            ["t5xxl_fp16.safetensors", "ltxv", "default"], title="Load CLIP (set your encoder)", color=IN)
vid = node("LoadVideo", (40, 290), (320, 300), [], [("VIDEO", "VIDEO")], ["input.mp4"], title="Load Video", color=IN)
comp = node("GetVideoComponents", (400, 290), (240, 100), [("video", "VIDEO")],
            [("images", "IMAGE"), ("audio", "AUDIO"), ("fps", "FLOAT")])
link(vid, 0, comp, "video")

# ---- krish prep
prep = node("KrishLTXFramePrep", (680, 290), (320, 190), [("images", "IMAGE")],
            [("images", "IMAGE"), ("conform_info", "KRISH_CONFORM"), ("num_frames", "INT"),
             ("width", "INT"), ("height", "INT")],
            [8, 32, "repeat_last", "reflect", "center"], title="Krish LTX Frame Prep", color=KRISH)
link(comp, 0, prep, "images")

enc = node("VAEEncode", (1040, 290), (200, 70), [("pixels", "IMAGE"), ("vae", "VAE")], [("LATENT", "LATENT")])
link(prep, 0, enc, "pixels")
link(ckpt, 2, enc, "vae")

# ---- prompts
pos = node("CLIPTextEncode", (400, 0), (420, 130), [("clip", "CLIP")], [("CONDITIONING", "CONDITIONING")],
           ["describe the edit you want, keep the original motion and camera"], title="Positive prompt")
neg = node("CLIPTextEncode", (400, 150), (420, 110), [("clip", "CLIP")], [("CONDITIONING", "CONDITIONING")],
           ["worst quality, inconsistent motion, blurry, jittery, distorted, watermark"], title="Negative prompt")
link(clip, 0, pos, "clip")
link(clip, 0, neg, "clip")

cond = node("LTXVConditioning", (860, 40), (260, 100),
            [("positive", "CONDITIONING"), ("negative", "CONDITIONING"), ("frame_rate", "FLOAT")],
            [("positive", "CONDITIONING"), ("negative", "CONDITIONING")], [25.0])
link(pos, 0, cond, "positive")
link(neg, 0, cond, "negative")
link(comp, 2, cond, "frame_rate")

# ---- sampling
sched = node("LTXVScheduler", (1280, 0), (280, 170), [("latent", "LATENT")], [("SIGMAS", "SIGMAS")],
             [30, 2.05, 0.95, True, 0.1])
link(enc, 0, sched, "latent")
split = node("SplitSigmasDenoise", (1280, 200), (280, 80), [("sigmas", "SIGMAS")],
             [("high_sigmas", "SIGMAS"), ("low_sigmas", "SIGMAS")], [0.5], title="Denoise (lower = keep more original)")
link(sched, 0, split, "sigmas")
ksel = node("KSamplerSelect", (1280, 310), (280, 70), [], [("SAMPLER", "SAMPLER")], ["euler"])
samp = node("SamplerCustom", (1600, 0), (300, 330),
            [("model", "MODEL"), ("positive", "CONDITIONING"), ("negative", "CONDITIONING"),
             ("sampler", "SAMPLER"), ("sigmas", "SIGMAS"), ("latent_image", "LATENT")],
            [("output", "LATENT"), ("denoised_output", "LATENT")], [True, 42, "randomize", 3.0])
link(ckpt, 0, samp, "model")
link(cond, 0, samp, "positive")
link(cond, 1, samp, "negative")
link(ksel, 0, samp, "sampler")
link(split, 1, samp, "sigmas")
link(enc, 0, samp, "latent_image")

dec = node("VAEDecode", (1940, 0), (200, 70), [("samples", "LATENT"), ("vae", "VAE")], [("IMAGE", "IMAGE")])
link(samp, 0, dec, "samples")
link(ckpt, 2, dec, "vae")

# ---- krish restore + parity
rest = node("KrishLTXFrameRestore", (1940, 120), (320, 190),
            [("generated", "IMAGE"), ("conform_info", "KRISH_CONFORM")],
            [("images", "IMAGE"), ("report", "STRING")], ["nearest", "bilinear"],
            title="Krish LTX Frame Restore", color=KRISH)
link(dec, 0, rest, "generated")
link(prep, 1, rest, "conform_info")

par = node("KrishFrameParityCheck", (2300, 120), (300, 120),
           [("source", "IMAGE"), ("result", "IMAGE")], [("result", "IMAGE"), ("report", "STRING")], ["error"],
           title="Krish Frame Parity Check", color=KRISH)
link(comp, 0, par, "source")
link(rest, 0, par, "result")

mk = node("CreateVideo", (2640, 120), (240, 100),
          [("images", "IMAGE"), ("fps", "FLOAT"), ("audio", "AUDIO")], [("VIDEO", "VIDEO")], [25.0],
          title="Create Video (source fps + audio)")
link(par, 0, mk, "images")
link(comp, 2, mk, "fps")
link(comp, 1, mk, "audio")

save = node("SaveVideo", (2920, 120), (320, 300), [("video", "VIDEO")], [],
            ["video/krish_ltx_exact", "auto", "auto"], color=OUT)
link(mk, 0, save, "video")

show1 = node("PreviewAny", (2300, 280), (300, 100), [("source", "*")], [], [], title="Restore report")
link(rest, 1, show1, "source")
show2 = node("PreviewAny", (2300, 400), (300, 100), [("source", "*")], [], [], title="Parity report")
link(par, 1, show2, "source")

wf = {"id": "krish-ltx-exact-frames-v2v", "revision": 0, "last_node_id": _nid[0], "last_link_id": _lid[0],
      "nodes": nodes, "links": links, "groups": [], "config": {}, "extra": {}, "version": 0.4}

out = pathlib.Path(__file__).parents[1] / "workflows" / "krish_ltx_exact_frames_v2v.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(wf, indent=2))

# self-check: every link referenced consistently on both ends
by_id = {n["id"]: n for n in nodes}
for lid, s, ss, d, ds, t in links:
    assert lid in by_id[s]["outputs"][ss]["links"], lid
    assert by_id[d]["inputs"][ds]["link"] == lid, lid
for n in nodes:
    for i in n["inputs"]:
        assert i["link"] is not None, (n["type"], i["name"])
print(f"wrote {out} ({len(nodes)} nodes, {len(links)} links)")

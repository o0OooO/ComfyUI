#!/usr/bin/env python3
"""MiniMax H3 t2va: text -> 5s 768p video with native 32kHz stereo audio.

Runs on a single 48GB L40S, which only fits because ComfyUI swaps the text
encoder out after encoding. Writes the API-format workflow to
user/default/workflows/minimax_h3/ and optionally submits it. Node/input names
are validated against a live /object_info so the JSON can't silently drift from
the installed ComfyUI.

Defaults to the 8-step Turbo LoRA -- see PRESETS for why that and not 4-step.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

SERVER = "http://127.0.0.1:8188"
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))          # .../ComfyUI
OUT_DIR = os.path.join(REPO, "user/default/workflows/minimax_h3")

# 1344x768 is H3's native canvas (768 short edge, 768*1344 area cap).
WIDTH, HEIGHT = 1344, 768
# length must sit on the 17k+5 grid: 124 frames @ 24fps = ~5.17s
LENGTH = 124

# t2va's pair; ref2va uses res_multistep/beta instead (see gen_minimax_h3_ref2va.py).
SAMPLER = "euler"
SCHEDULER = "simple"

# lightx2v's distilled LoRAs, each with its own step count and shift contract.
#
# 8step is the default because the 4-step LoRA measurably drops the prompt's
# shallow depth of field: Laplacian variance on the out-of-focus background strip
# came out 31.0 for 8step and 35.0 for the 30-step baseline, against 107.1 for
# 4step -- the background renders nearly in focus and skin goes waxy. That is a
# property of the LoRA, not of the step count: raising 4step's shift to 12 only
# got to 78.0, and driving 4step at 8 steps made it *worse* (116.8) since it is
# bound to its trained schedule. Joint audio splits the same way (4step sits at
# -31..-33 dBFS mean regardless of schedule; 8step and base at -18..-21).
# So 8step ~= the 30-step baseline at 292s instead of 1327s. Keep 4step for
# previews and blocking only.
PRESETS = {
    # name:  (lora file or None, steps, shift_video, shift_audio)
    "8step": ("minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors", 8, 12.0, 3.0),
    "4step": ("minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors", 4, 6.0, 3.0),
    "base": (None, 30, 12.0, 3.0),
}

# Two quant pairs. int8_convrot has native sm_89 kernels (int8_tensorwise), so it
# samples ~8% faster per step; w4a8_mixed is half the bytes, so it wins cold start
# by staging 27.7GB instead of 45.9GB. Warm they tie at ~177s on 4 steps, and at
# 8 steps int8 wins by ~32s. Picture quality is equivalent at a fixed seed.
QUANTS = {
    "int8": ("minimax_h3_fl2va_pruned_int8_convrot.safetensors",
             "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"),
    "w4a8": ("minimax_h3_fl2va_pruned_w4a8_mixed.safetensors",
             "qwen3vl_32b_minimax_h3_w4a8_mixed.safetensors"),
}

VAE_VIDEO = "minimax_h3_video_vae_fp16.safetensors"
VAE_AUDIO = "minimax_h3_audio_vae_fp32.safetensors"

# H3 jointly models picture and sound, so the audio has to be described in the
# same paragraph as the visuals -- not appended as an afterthought.
PROMPT = (
    "A lone street musician plays a saxophone under a flickering neon sign on a "
    "rain-slicked city street at night. Slow dolly-in, shallow depth of field, "
    "cinematic film grain. Audio: a warm melancholic saxophone melody, distant "
    "traffic hum, light rain pattering on pavement."
)


def fetch_object_info():
    cache = "/tmp/comfy_object_info.json"
    try:
        with urllib.request.urlopen(SERVER + "/object_info", timeout=30) as r:
            oi = json.load(r)
        with open(cache, "w") as f:
            json.dump(oi, f)
        return oi
    except Exception as e:
        if not os.path.exists(cache):
            raise SystemExit(
                f"can't reach /object_info ({e}). start ComfyUI first:\n"
                f"  bash user/default/scripts/start_comfy.sh")
        print(f"! /object_info unavailable ({e}), using cache {cache}")
        return json.load(open(cache))


def build(preset="8step", quant="int8", prompt=PROMPT, seed=42,
          steps=None, shift_video=None, shift_audio=None, prefix=None):
    """API-format t2va graph."""
    lora, p_steps, p_sv, p_sa = PRESETS[preset]
    steps = p_steps if steps is None else steps
    shift_video = p_sv if shift_video is None else shift_video
    shift_audio = p_sa if shift_audio is None else shift_audio
    unet_name, clip_name = QUANTS[quant]

    # the LoRA patches the quantized weights in place, so the matmuls stay
    # quantized and only the patched rows get requantized per step
    model_src = "lora" if lora else "unet"

    wf = {
        "unet": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": unet_name, "weight_dtype": "default"},
        },
        "clip": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": clip_name, "type": "minimax"},
        },
        "vae": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": VAE_VIDEO},
        },
        "audio_vae": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": VAE_AUDIO},
        },
        # t2va: no keyframes connected, so this is pure text-to-audio-video
        "cond": {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": {
                "clip": ["clip", 0],
                "vae": ["vae", 0],
                "prompt": prompt,
                "width": WIDTH,
                "height": HEIGHT,
                "length": LENGTH,
            },
        },
        # shift_video drives the sampler schedule; the DiT derives audio's from it
        "shift": {
            "class_type": "MiniMaxH3SigmaShift",
            "inputs": {"model": [model_src, 0],
                       "shift_video": shift_video, "shift_audio": shift_audio},
        },
        "guider": {
            "class_type": "BasicGuider",
            "inputs": {"model": ["shift", 0], "conditioning": ["cond", 0]},
        },
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "sampler": {"class_type": "KSamplerSelect",
                    "inputs": {"sampler_name": SAMPLER}},
        "sigmas": {
            "class_type": "BasicScheduler",
            "inputs": {"model": ["shift", 0], "scheduler": SCHEDULER,
                       "steps": steps, "denoise": 1.0},
        },
        "sample": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {"noise": ["noise", 0], "guider": ["guider", 0],
                       "sampler": ["sampler", 0], "sigmas": ["sigmas", 0],
                       "latent_image": ["cond", 1]},
        },
        # the sampled pack is a NestedTensor pair; each VAE pulls its own stream
        "decode_video": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]},
        },
        "decode_audio": {
            "class_type": "VAEDecodeAudio",
            "inputs": {"samples": ["sample", 0], "vae": ["audio_vae", 0]},
        },
        "video": {
            "class_type": "CreateVideo",
            "inputs": {"images": ["decode_video", 0], "fps": 24.0,
                       "audio": ["decode_audio", 0]},
        },
        "save": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["video", 0],
                # only spell out the knobs that deviate from the preset, so the
                # shipped default gets a clean name and A/B runs stay labelled
                "filename_prefix": prefix or (
                    f"video/MiniMax_H3_{preset}_{quant}"
                    + (f"_{steps}step" if steps != p_steps else "")
                    + (f"_sv{shift_video:g}" if shift_video != p_sv else "")
                    + (f"_sa{shift_audio:g}" if shift_audio != p_sa else "")),
                "format": "auto",
                "codec": "auto",
            },
        },
    }
    if lora:
        wf["lora"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": ["unet", 0], "lora_name": lora,
                       "strength_model": 1.0},
        }
    return wf


def validate(wf, oi):
    """Fail loudly on unknown node types, inputs, or combo values."""
    errs = []
    for nid, node in wf.items():
        ct = node["class_type"]
        if ct not in oi:
            errs.append(f"{nid}: unknown node {ct}")
            continue
        spec = oi[ct]["input"]
        known = dict(spec.get("required", {}))
        known.update(spec.get("optional", {}))
        for k, v in node["inputs"].items():
            base = k.split(".")[0]
            if k not in known and base not in known:
                errs.append(f"{nid} ({ct}): unknown input {k}")
                continue
            entry = known.get(k)
            if entry is None or not isinstance(v, str):
                continue
            # combos are either ["COMBO", {options: [...]}] or a bare value list;
            # dynamic combos carry dict entries keyed by "key"
            opts = entry[1].get("options") if len(entry) > 1 and isinstance(entry[1], dict) else None
            if opts is None and isinstance(entry[0], list):
                opts = entry[0]
            if not opts:
                continue
            allowed = [o["key"] if isinstance(o, dict) else o for o in opts]
            if v not in allowed:
                errs.append(f"{nid} ({ct}): {k}={v!r} not in {allowed[:8]}")
    if errs:
        raise SystemExit("workflow validation failed:\n  " + "\n  ".join(errs))


# Anything not in here is a link slot rather than a widget.
WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}


def _is_widget(decl_type):
    return (isinstance(decl_type, list)                     # bare combo list
            or decl_type in WIDGET_TYPES
            or str(decl_type).startswith("COMFY_DYNAMICCOMBO"))


def _topo(wf):
    """Node keys ordered so every producer precedes its consumers."""
    order, seen = [], set()

    def visit(k):
        if k in seen:
            return
        seen.add(k)
        for v in wf[k]["inputs"].values():
            if isinstance(v, list) and len(v) == 2 and v[0] in wf:
                visit(v[0])
        order.append(k)

    for k in wf:
        visit(k)
    return order


NOTE = """## 本机改动（对着官方 t2v 模板）

1. **TE 换成 int8_convrot**，不是模板的 `nvfp4_awq` —— L40S 是 sm_89，NVFP4 只能模拟执行
2. **加了 `LoraLoaderModelOnly`**：`minimax_h3_fl2v_turbo_8step_v1.0`，30 步 → 8 步
3. **加了 `MiniMaxH3SigmaShift`** 12.0 / 3.0（8 步 LoRA 的 shift 契约）
4. **`euler` / `simple`**，不是模板的 `res_multistep`（那是 ref2va 用的）
5. **1344×768 / 124 帧写死**，没走 `ResolutionSelector` 子图

⚠️ 别换成 4 步那个 LoRA：实测背景虚化会塌（背景 Laplacian 方差 107 vs 8 步的 31，
30 步基线是 35），prompt 里的 shallow depth of field 出不来。细节见
`user/default/docs/MINIMAX_H3_GUIDE.md`。"""


def to_ui(wf, oi):
    """Convert the API graph into a flat UI-format graph for the sidebar.

    Generated, not hand-edited: widget order and defaults come from the live
    /object_info, so this can't drift from the installed ComfyUI either.
    """
    keys = _topo(wf)
    nid = {k: i + 1 for i, k in enumerate(keys)}

    # column = longest path from a source, so producers sit left of consumers
    depth = {}
    for k in keys:
        srcs = [v[0] for v in wf[k]["inputs"].values()
                if isinstance(v, list) and len(v) == 2 and v[0] in wf]
        depth[k] = max((depth[s] + 1 for s in srcs), default=0)

    nodes, links, link_id = [], [], 0
    col_rows = {}
    # producer key/slot -> list of link ids, filled in as consumers are wired
    out_links = {}

    for k in keys:
        ct = wf[k]["class_type"]
        spec = oi[ct]["input"]
        decls = list(spec.get("required", {}).items()) + \
            list(spec.get("optional", {}).items())

        inputs, widgets = [], []
        for name, entry in decls:
            decl_type = entry[0]
            meta = entry[1] if len(entry) > 1 and isinstance(entry[1], dict) else {}
            val = wf[k]["inputs"].get(name)
            if _is_widget(decl_type):
                if val is None:
                    # not in the API graph, so fall back to the widget's own
                    # default -- combos often declare none, then it's option 0
                    val = meta.get("default")
                    if val is None:
                        opts = meta.get("options") or (
                            decl_type if isinstance(decl_type, list) else [])
                        if opts:
                            val = opts[0]["key"] if isinstance(opts[0], dict) else opts[0]
                widgets.append(val)
                if meta.get("control_after_generate"):
                    widgets.append("fixed")
            elif isinstance(val, list) and len(val) == 2:
                link_id += 1
                links.append([link_id, nid[val[0]], val[1], nid[k],
                              len(inputs), decl_type])
                out_links.setdefault((val[0], val[1]), []).append(link_id)
                inputs.append({"name": name, "type": decl_type, "link": link_id})
            else:
                # unconnected link slot (e.g. first_frame) -- keep it visible
                inputs.append({"name": name, "type": decl_type, "link": None})

        out_names = oi[ct].get("output_name") or oi[ct]["output"]
        outputs = [{"name": out_names[i] if i < len(out_names) else t,
                    "type": t, "links": []}
                   for i, t in enumerate(oi[ct]["output"])]

        d = depth[k]
        row = col_rows.get(d, 0)
        col_rows[d] = row + 1
        h = 60 + 26 * len(widgets) + 22 * len(inputs)
        nodes.append({
            "id": nid[k], "type": ct,
            "pos": [d * 420, row * 300],
            "size": [380, h],
            "flags": {}, "order": keys.index(k), "mode": 0,
            "inputs": inputs, "outputs": outputs,
            "properties": {"Node name for S&R": ct},
            "widgets_values": widgets,
        })

    by_id = {n["id"]: n for n in nodes}
    for (src_k, slot), lids in out_links.items():
        by_id[nid[src_k]]["outputs"][slot]["links"] = lids

    max_col = max(depth.values())
    nodes.append({
        "id": len(keys) + 1, "type": "MarkdownNote",
        "pos": [(max_col + 1) * 420, 0], "size": [430, 420],
        "flags": {}, "order": len(keys), "mode": 0,
        "inputs": [], "outputs": [],
        "properties": {}, "widgets_values": [NOTE], "color": "#432",
        "bgcolor": "#653",
    })

    return {
        "id": "8a1c7d34-2b96-4e07-9f51-6d0ab3e17c42",
        "revision": 0,
        "last_node_id": len(keys) + 1,
        "last_link_id": link_id,
        "nodes": nodes, "links": links, "groups": [],
        "config": {}, "extra": {}, "version": 0.4,
    }


def check_ui_matches_api(ui, wf, oi):
    """Every effective widget value in the UI graph must equal the API graph's."""
    errs = []
    byid = {n["id"]: n for n in ui["nodes"]}
    linked = {(l[3], l[4]) for l in ui["links"]}
    keys = _topo(wf)
    nid = {k: i + 1 for i, k in enumerate(keys)}
    for k in keys:
        ct = wf[k]["class_type"]
        n = byid[nid[k]]
        spec = oi[ct]["input"]
        decls = list(spec.get("required", {}).items()) + \
            list(spec.get("optional", {}).items())
        wi = 0
        for name, entry in decls:
            meta = entry[1] if len(entry) > 1 and isinstance(entry[1], dict) else {}
            if _is_widget(entry[0]):
                got = n["widgets_values"][wi]
                wi += 1 + bool(meta.get("control_after_generate"))
                want = wf[k]["inputs"].get(name)
                if want is not None and got != want:
                    errs.append(f"{k} ({ct}).{name}: ui={got!r} api={want!r}")
            elif isinstance(wf[k]["inputs"].get(name), list):
                slot = next((i for i, s in enumerate(n["inputs"])
                             if s["name"] == name), None)
                if slot is None or (n["id"], slot) not in linked:
                    errs.append(f"{k} ({ct}).{name}: api links it, ui does not")
    if errs:
        raise SystemExit("UI/API mismatch:\n  " + "\n  ".join(errs))


def post(path, payload):
    req = urllib.request.Request(
        SERVER + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def get(path):
    return json.loads(urllib.request.urlopen(SERVER + path, timeout=60).read())


def run(wf, label):
    sig = wf["sigmas"]["inputs"]
    sh = wf["shift"]["inputs"]
    print(f"[{label}] submitting: {WIDTH}x{HEIGHT}, {LENGTH}f (~{LENGTH/24:.1f}s), "
          f"{sig['steps']} steps, shift {sh['shift_video']}/{sh['shift_audio']}, "
          f"{SAMPLER}/{SCHEDULER}")
    try:
        pid = post("/prompt", {"prompt": wf})["prompt_id"]
    except urllib.error.HTTPError as e:
        print(f"[{label}] SUBMIT FAILED:", e.read().decode()[:4000])
        return None
    print(f"[{label}] prompt_id:", pid)
    start = time.time()
    while True:
        time.sleep(10)
        hist = get(f"/history/{pid}")
        if pid in hist:
            entry = hist[pid]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                print(f"[{label}] FAILED after %.0fs" % (time.time() - start))
                for m in status.get("messages", []):
                    print("  ", json.dumps(m)[:2000])
                return None
            print(f"[{label}] DONE in %.0fs" % (time.time() - start))
            out = []
            for _, o in entry.get("outputs", {}).items():
                for v in o.get("images", []) + o.get("video", []):
                    print("   output:", v)
                    out.append(v)
            return out
        q = get("/queue")
        print("   ... %.0fs (running=%d)" % (time.time() - start,
                                            len(q.get("queue_running", []))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="8step", choices=list(PRESETS),
                    help="8step = shipped default (~= 30-step quality at 292s); "
                         "4step = preview only, loses depth of field; base = 30 steps")
    ap.add_argument("--quant", default="int8", choices=list(QUANTS),
                    help="int8 for a resident server, w4a8 for one-off cold starts")
    ap.add_argument("--steps", type=int, default=None,
                    help="override the preset's step count (A/B only)")
    ap.add_argument("--shift-video", type=float, default=None)
    ap.add_argument("--shift-audio", type=float, default=None)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--prompt-file", default=None)
    # Same prompt + same seed hits ComfyUI's output cache and returns in ~0s,
    # which silently invalidates any timing comparison -- vary one when measuring.
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--write-only", action="store_true",
                    help="write the workflow JSON without submitting")
    args = ap.parse_args()

    prompt = args.prompt or PROMPT
    if args.prompt_file:
        prompt = open(args.prompt_file).read()

    wf = build(preset=args.preset, quant=args.quant, prompt=prompt,
               seed=args.seed, steps=args.steps,
               shift_video=args.shift_video, shift_audio=args.shift_audio)
    oi = fetch_object_info()
    validate(wf, oi)

    # Only the shipped default is tracked in git; A/B variants would churn it.
    if (args.preset, args.quant) == ("8step", "int8") and args.steps is None \
            and args.shift_video is None and args.shift_audio is None:
        os.makedirs(OUT_DIR, exist_ok=True)
        ui = to_ui(wf, oi)
        check_ui_matches_api(ui, wf, oi)
        for name, doc in (("t2va_8step.api.json", wf),
                          ("t2va_8step_ui.json", ui)):
            path = os.path.join(OUT_DIR, name)
            with open(path, "w") as f:
                json.dump(doc, f, indent=2, ensure_ascii=False)
            print("wrote", os.path.relpath(path, REPO))

    if args.write_only:
        return 0
    return 0 if run(wf, f"{args.preset}/{args.quant}") else 1


if __name__ == "__main__":
    sys.exit(main())

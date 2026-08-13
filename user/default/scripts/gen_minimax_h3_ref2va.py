#!/usr/bin/env python3
"""MiniMax H3 ref2va: multi-reference (images / videos / audio) -> 5s 768p video + audio.

ref2va needs its OWN transformer (upstream ships transformer/ and transformer_ref/
side by side). The fl2va file used by gen_minimax_h3_t2va.py cannot do this task.

Writes the API-format workflow to user/default/workflows/minimax_h3/ and optionally
submits it. Node/input names are validated against a live /object_info so the JSON
can't silently drift from the installed ComfyUI.
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

# The community hybrid keeps fl2va's higher-fidelity weights and only swaps the
# later blocks' adaln_proj (the reference pathway) — same int8_convrot layout,
# drop-in. Verified byte-exact: blocks 0-24 adaln_proj = fl2va, 25-49 = ref2va,
# everything else = fl2va.
#
# Head-to-head at seed 42 (see workflows/minimax_h3/README.md) picked hybrid, so
# only it is kept on disk. "official" needs a ~4min re-download first:
#   hf download Comfy-Org/MiniMax-H3 \
#     diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors
MODEL_HYBRID = "minimax_h3_hybrid_fl2va_ref2va_b25-49.safetensors"
MODEL_OFFICIAL = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"

TE = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
VAE_VIDEO = "minimax_h3_video_vae_fp16.safetensors"
VAE_AUDIO = "minimax_h3_audio_vae_fp32.safetensors"

WIDTH, HEIGHT = 1344, 768
LENGTH = 124          # 17k+5 grid @ 24fps = 5.17s
STEPS = 20            # official R2V template's value
SHIFT_VIDEO, SHIFT_AUDIO = 12.0, 3.0

# res_multistep + beta, not the euler/simple pair t2va uses: the official template
# notes beta/normal beats simple on reference-heavy prompts.
SAMPLER = "res_multistep"
SCHEDULER = "beta"

REF_IMAGES = ["h3ref_woman.png", "h3ref_truck.png"]

# The hybrid's joint audio comes out ~8dB quieter than the official ref2va's
# (measured: mean -32.2 vs -23.8 dBFS, peak -11.5 vs -4.3). Dialogue is present
# with the same dynamic range, just low, so a flat gain fixes it and there is
# enough headroom not to clip. 0 disables the node entirely.
AUDIO_GAIN_DB = 8

# The six-section ref2va format from the model card's
# docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md. Section order is fixed. <Subject N>
# is the reusable entity; <Picture N> is only the asset carrying it — keeping them
# separate is what lets you borrow a character without inheriting its composition.
PROMPT = """subject_definitions:
<Subject 1> is the young woman in <Picture 1>, with long auburn hair lit warm by low sun, a cream-coloured pintucked prairie blouse with a crocheted collar and rolled sleeves, a dark brown leather belt, and a long sand-coloured skirt.
<Subject 2> is the weathered red 1960s pickup truck in <Picture 2>, with rust blooming across the hood and front fender, a chrome grille and dual round headlights, a chrome side mirror, and dark steel wheels.
<Subject 3> is the open prairie at golden hour from <Picture 1> and <Picture 2>: a gravel road, flowering grassland on both sides, a distant wire fence, and a pale hazy sky.

summary:
[reference generation] The target video keeps <Subject 1> and <Subject 2> exactly as defined and stages a new single-shot scene in <Subject 3>: she walks to the truck's open driver door, leans on the frame, and speaks one line to camera. Neither reference's original framing or pose is reused.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - her facial identity, auburn hair, cream pintucked blouse with crocheted collar, brown leather belt, and long sand skirt are retained. Her pose, position in frame, and direction of gaze are newly generated.
<Subject 2> (appears in [Shot 1]): fully_preserved - the truck's red paint, rust pattern on hood and fender, chrome grille, dual round headlights, side mirror, and steel wheels are retained. Its camera angle and position are newly generated.
<Subject 3> (appears in [Shot 1]): partially_preserved - the gravel road, flowering grassland, wire fence, and golden-hour haze are retained as a setting; the exact composition of either reference is not reproduced.

detailed_description:
The target video is in realistic photographic style with anamorphic golden-hour light, shallow depth of field, and fine 35mm grain.
[Shot 1] A medium tracking shot follows <Subject 1>, the auburn-haired young woman in the cream pintucked blouse and long sand skirt, as she walks along the gravel road toward <Subject 2>, the rust-blotched red pickup, its driver door standing open. Warm low sun rakes across her left shoulder and flares softly off the truck's chrome grille. She reaches the open door, rests her right forearm along the window frame, and turns her face to camera. Wind lifts a few strands of her hair across her cheek. <Subject 1> (S1) says, in a warm unhurried voice with a slight rasp, <d>[English] Engine's dead. Guess we're walking.</d> Her lips close and she glances off toward the horizon with a faint resigned smile, tapping her fingers twice on the door frame as the camera settles.

overall_soundscape:
Steady dry prairie wind across open grassland, gravel crunching under her boots as she walks, a soft metallic creak from the open truck door, and faint insect chirr in the tall grass.

non_diegetic_music:
A sparse, slow acoustic guitar figure with light room reverb, entering quietly under the dialogue and holding through the final frame."""


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


def build(model_name, ref_images=REF_IMAGES, prompt=PROMPT, seed=42,
          steps=STEPS, ref_image_size="match", prefix=None,
          audio_gain=AUDIO_GAIN_DB):
    """API-format ref2va graph. Autogrow slots use flat dotted keys."""
    refs = {f"ref_images.ref_image_{i}": [f"load_{i}", 0]
            for i in range(len(ref_images))}

    wf = {
        "unet": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": model_name, "weight_dtype": "default"},
        },
        "clip": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": TE, "type": "minimax"},
        },
        "vae": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": VAE_VIDEO},
        },
        "audio_vae": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": VAE_AUDIO},
        },
        "cond": {
            "class_type": "MiniMaxH3ReferenceToVideo",
            "inputs": {
                "clip": ["clip", 0],
                "vae": ["vae", 0],
                "audio_vae": ["audio_vae", 0],
                "prompt": prompt,
                "width": WIDTH,
                "height": HEIGHT,
                "length": LENGTH,
                "ref_image_size": ref_image_size,
                **refs,
            },
        },
        "shift": {
            "class_type": "MiniMaxH3SigmaShift",
            "inputs": {"model": ["unet", 0],
                       "shift_video": SHIFT_VIDEO, "shift_audio": SHIFT_AUDIO},
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
                       "audio": ["gain" if audio_gain else "decode_audio", 0]},
        },
        "save": {
            "class_type": "SaveVideo",
            "inputs": {"video": ["video", 0],
                       "filename_prefix": prefix or "video/MiniMax_H3_ref2va",
                       "format": "auto", "codec": "auto"},
        },
    }
    if audio_gain:
        wf["gain"] = {
            "class_type": "AudioAdjustVolume",
            "inputs": {"audio": ["decode_audio", 0], "volume": int(audio_gain)},
        }
    for i, name in enumerate(ref_images):
        wf[f"load_{i}"] = {"class_type": "LoadImage",
                           "inputs": {"image": name}}
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


def post(path, payload):
    req = urllib.request.Request(
        SERVER + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def get(path):
    return json.loads(urllib.request.urlopen(SERVER + path, timeout=60).read())


def run(wf, label):
    print(f"[{label}] submitting: {WIDTH}x{HEIGHT}, {LENGTH}f "
          f"(~{LENGTH/24:.1f}s), {STEPS} steps, {SAMPLER}/{SCHEDULER}")
    try:
        pid = post("/prompt", {"prompt": wf})["prompt_id"]
    except urllib.error.HTTPError as e:
        print(f"[{label}] SUBMIT FAILED:", e.read().decode()[:4000])
        return None
    print(f"[{label}] prompt_id:", pid)
    start = time.time()
    while True:
        time.sleep(15)
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
    ap.add_argument("--model", default="hybrid",
                    choices=["hybrid", "official"],
                    help="which ref2va checkpoint to use (official is not on disk)")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ref-image-size", default="match", choices=["match", "max"])
    ap.add_argument("--audio-gain", type=int, default=AUDIO_GAIN_DB,
                    help="dB applied to the generated audio (0 = no gain node)")
    ap.add_argument("--ref", action="append", default=None,
                    help="reference image filename in input/ (repeatable, <=9)")
    ap.add_argument("--prompt-file", default=None,
                    help="read the six-section prompt from a file")
    ap.add_argument("--write-only", action="store_true",
                    help="write the workflow JSON without submitting")
    args = ap.parse_args()

    model = MODEL_OFFICIAL if args.model == "official" else MODEL_HYBRID
    refs = args.ref or REF_IMAGES
    if len(refs) > 9:
        raise SystemExit("ref2va accepts at most 9 reference images")
    prompt = PROMPT
    if args.prompt_file:
        prompt = open(args.prompt_file).read()

    wf = build(model, ref_images=refs, prompt=prompt, seed=args.seed,
               steps=args.steps, ref_image_size=args.ref_image_size,
               prefix=f"video/MiniMax_H3_ref2va_{args.model}",
               audio_gain=args.audio_gain)
    validate(wf, fetch_object_info())

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"ref2va_{args.model}.api.json")
    with open(path, "w") as f:
        json.dump(wf, f, indent=2, ensure_ascii=False)
    print("wrote", os.path.relpath(path, REPO))

    if args.write_only:
        return 0
    return 0 if run(wf, args.model) else 1


if __name__ == "__main__":
    sys.exit(main())

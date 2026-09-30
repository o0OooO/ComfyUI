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
# The seed-42 head-to-head below picked hybrid, but that was between two
# *int8_convrot* files. Defaults now point at the **w4a8 tier** instead, and not
# for speed — with clean VRAM the two tiers run at the same s/it. It is about
# surviving: H3 stages all three weights together, and the int8 tier totals
# 50842MB > the card's 45460MiB. The overflow lives in CPU pinned memory, so
#   (a) any other resident model (SenseNova 34GB / WAN21_Vace 33GB) pushes the
#       transformer out too, and every sampling step re-stages 20GB
#       → 173.78 s/it, i.e. 58 min for one 5s clip;
#   (b) pinned-resident + a 124-frame VAE decode drives comfy RSS to 53-59GB on
#       a 61GB box with **Swap 0** → killed by the kernel OOM killer.
# w4a8 totals 32632MB (12GB headroom, RSS peak 35GB). Measured 2026-08-20; see
# SceneForge services/worker-ai/app/providers/comfyui_graphs/minimax_h3.py:51.
# Same-seed identity between official-w4a8 and hybrid came out a tie.
MODEL_HYBRID = "minimax_h3_hybrid_fl2va_ref2va_b25-49.safetensors"
MODEL_OFFICIAL = "minimax_h3_ref2va_pruned_w4a8_mixed.safetensors"

TE = "qwen3vl_32b_minimax_h3_w4a8_mixed.safetensors"
VAE_VIDEO = "minimax_h3_video_vae_fp16.safetensors"
VAE_AUDIO = "minimax_h3_audio_vae_fp32.safetensors"

WIDTH, HEIGHT = 1344, 768
LENGTH = 124          # 17k+5 grid @ 24fps = 5.17s
STEPS = 20            # official R2V template's value
SHIFT_VIDEO, SHIFT_AUDIO = 12.0, 3.0

# What actually costs VRAM is the patchified sequence length, not pixels·frames.
# From nodes_minimax_h3.py + model.py: video_latent_t(n) = ((n-5)//17)*5+2, the
# spatial grid is (w//16)x(h//16), and patchify_video uses patch_size (1,2,2):
#
#     rows = video_latent_t(length) * (w//32) * (h//32)          [video]
#          + n_refs * (w//32) * (h//32)                          [refs, match mode]
#
# Measured on a 48GB L40S with **clean VRAM** (2026-09-02), all 3 refs:
#     frame_rows  rows    peak VRAM    verdict
#       1008     40320    32987 MiB    ok  (1344x768 x 124f, 44.72 s/it, 18:30)
#        405     36450    ~33 GB       ok  ( 864x480 x 294f, 38.89 s/it, 16:45)
#        405     44550    31643 MiB    ok  ( 864x480 x 362f, 49.46 s/it, 20:15,
#                                           comfy RSS 42G on the 362-frame decode)
#       1008     90720      —          OOM (1344x768 x 294f, 2026-08-31, 4/4)
#
# Note rows is a **conservative proxy, not a memory model**: 44550 rows used
# *less* VRAM than 40320 because what really costs peak memory is the per-frame
# spatial size (frame_rows), not the total. So the budget is split **by
# frame_rows into two tiers**, each capped at its own largest measured pass,
# rather than taking 44550 globally: a global 44550 would let a number measured
# at 480p wave through 1344x768 x 141f (45360 rows w/ 3 refs), an unrun point,
# while the high-spatial tier's only passing point is 40320. The split is at 405
# (= the measured low tier); canvases in between (720p -> frame_rows 880) are
# treated as high — better to lose 0.7s than to gamble 20 minutes of compute.
# Raising either tier takes one more measurement, never a guess: guessing high
# costs an OOM and the whole run.
#
# Two traps this catches that the 17k+5 grid check cannot:
#   * canvas and length are independent knobs, so 1344x768 x 294f is a perfectly
#     legal request — and it is the 2026-08-31 online failure that got blamed on
#     "768 short edge is too big". 768 is fine (row 1); 768 *plus* 12s is not.
#   * ref_image_size="match" scales every reference to the generation's pixel
#     area, so each extra ref costs a full frame's worth of rows at 768p —
#     2.5x what it costs at 480p. References stop being free on the high tier.
LO_FRAME_ROWS = 405     # tier split = the measured low-spatial canvas (864x480)
MAX_ROWS_LO = 44550     # largest measured pass at frame_rows <= 405
MAX_ROWS_HI = 40320     # largest measured pass above it


def video_latent_t(length):
    """Temporal latent depth. Mirrors nodes_minimax_h3.py:40."""
    return 2 if length <= 5 else ((length - 5) // 17) * 5 + 2


def frame_rows_of(width, height):
    """Patch rows in one frame: the (w//16)x(h//16) grid halved on each axis."""
    return (width // 32) * (height // 32)


def row_budget(width, height):
    """Row budget for this canvas — see the measured table above for the tiers."""
    return MAX_ROWS_LO if frame_rows_of(width, height) <= LO_FRAME_ROWS else MAX_ROWS_HI


def latent_rows(width, height, length, n_refs):
    """Patchified sequence length fed to the DiT (video rows + reference rows)."""
    fr = frame_rows_of(width, height)
    return video_latent_t(length) * fr + n_refs * fr

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
          audio_gain=AUDIO_GAIN_DB, te=TE,
          width=WIDTH, height=HEIGHT, length=LENGTH):
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
            "inputs": {"clip_name": te, "type": "minimax"},
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
                "width": width,
                "height": height,
                "length": length,
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
    c = wf["cond"]["inputs"]
    print(f"[{label}] submitting: {c['width']}x{c['height']}, {c['length']}f "
          f"(~{c['length']/24:.2f}s), {wf['sigmas']['inputs']['steps']} steps, "
          f"{SAMPLER}/{SCHEDULER}, seed={wf['noise']['inputs']['noise_seed']}\n"
          f"[{label}] unet={wf['unet']['inputs']['unet_name']}\n"
          f"[{label}] te={wf['clip']['inputs']['clip_name']}\n"
          f"[{label}] refs=" + ", ".join(
              wf[f"load_{i}"]["inputs"]["image"]
              for i in range(len([k for k in c if k.startswith("ref_images.")]))))
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
    ap.add_argument("--model", default="official",
                    choices=["hybrid", "official"],
                    help="ref2va checkpoint. 'official' = pruned w4a8 (default, "
                         "ties hybrid on identity, 8GB less staged). 'hybrid' is "
                         "int8_convrot, so it stages ~4GB more than official; "
                         "still fits next to the w4a8 text encoder, but do not "
                         "pair it with the int8 TE (see the note on TE above).")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ref-image-size", default="match", choices=["match", "max"])
    ap.add_argument("--audio-gain", type=int, default=AUDIO_GAIN_DB,
                    help="dB applied to the generated audio (0 = no gain node)")
    ap.add_argument("--ref", action="append", default=None,
                    help="reference image filename in input/ (repeatable, <=9)")
    ap.add_argument("--prompt-file", default=None,
                    help="read the six-section prompt from a file")
    ap.add_argument("--width", type=int, default=WIDTH)
    ap.add_argument("--height", type=int, default=HEIGHT)
    ap.add_argument("--length", type=int, default=LENGTH,
                    help="frames @24fps, must sit on the 17k+5 grid (124/141/158/...)")
    ap.add_argument("--unet", default=None,
                    help="override the checkpoint filename (bypasses --model)")
    ap.add_argument("--te", default=TE, help="text encoder filename")
    ap.add_argument("--prefix", default=None,
                    help="SaveVideo filename_prefix (default video/MiniMax_H3_ref2va_<model>)")
    ap.add_argument("--force", action="store_true",
                    help="submit even if the latent-row budget is exceeded")
    ap.add_argument("--write-only", action="store_true",
                    help="write the workflow JSON without submitting")
    ap.add_argument("--tag", default=None,
                    help="suffix for the written api.json filename")
    args = ap.parse_args()

    if (args.length - 5) % 17:
        raise SystemExit(
            f"--length {args.length} is off the 17k+5 grid; "
            f"nearest are {5 + 17*((args.length-5)//17)} and {5 + 17*((args.length-5)//17 + 1)}")

    model = args.unet or (MODEL_OFFICIAL if args.model == "official" else MODEL_HYBRID)
    refs = args.ref or REF_IMAGES
    if len(refs) > 9:
        raise SystemExit("ref2va accepts at most 9 reference images")

    rows = latent_rows(args.width, args.height, args.length, len(refs))
    budget = row_budget(args.width, args.height)
    frame_rows = frame_rows_of(args.width, args.height)
    print(f"latent rows: {rows} / {budget} budget "
          f"({args.width}x{args.height} = {frame_rows} rows/frame, "
          f"{args.length}f, {len(refs)} refs)")
    if rows > budget and not args.force:
        fits = 5 + 17 * max(0, ((budget // frame_rows) - len(refs) - 2) // 5)
        raise SystemExit(
            f"{rows} latent rows is {100 * rows / budget - 100:.0f}% over "
            f"the {budget} budget for this canvas tier, which is the largest "
            f"configuration actually measured to fit — not a proven ceiling. "
            f"Attention is "
            f"quadratic in rows, and 2.25x over (1344x768 x 294f) is a confirmed "
            f"OOM; modest overshoots are simply untested.\n"
            f"  keep the canvas, cut frames:  --length {fits}\n"
            f"  or keep the length, drop to a smaller canvas.\n"
            f"  --force runs it anyway — do that with clean VRAM to move the "
            f"budget up on evidence instead of raising the constant on a guess.")
    prompt = PROMPT
    if args.prompt_file:
        prompt = open(args.prompt_file).read()

    wf = build(model, ref_images=refs, prompt=prompt, seed=args.seed,
               steps=args.steps, ref_image_size=args.ref_image_size,
               prefix=args.prefix or f"video/MiniMax_H3_ref2va_{args.model}",
               audio_gain=args.audio_gain, te=args.te,
               width=args.width, height=args.height, length=args.length)
    validate(wf, fetch_object_info())

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"ref2va_{args.tag or args.model}.api.json")
    with open(path, "w") as f:
        json.dump(wf, f, indent=2, ensure_ascii=False)
    print("wrote", os.path.relpath(path, REPO))

    if args.write_only:
        return 0
    return 0 if run(wf, args.model) else 1


if __name__ == "__main__":
    sys.exit(main())

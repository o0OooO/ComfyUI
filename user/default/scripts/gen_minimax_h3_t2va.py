#!/usr/bin/env python3
"""MiniMax H3 t2va smoke test: text -> 5s 768p video with native stereo audio.

Runs the int8_convrot quant pair (transformer 21GB + Qwen3-VL-32B TE 27GB) on a
single 48GB L40S, which only fits because ComfyUI swaps the TE out after
encoding. Submits via the /prompt API and polls until the video lands.

Defaults to the 4-step Turbo LoRA; pass --base for the original 30-step config,
--8step for the 8-step LoRA, --w4a8 for the smaller quant pair, or
--steps/--shift-audio/--shift-video to override any single knob when A/B-ing.
"""

import json
import sys
import time
import urllib.request

SERVER = "http://127.0.0.1:8188"

# 1344x768 is H3's native canvas (768 short edge, 768*1344 area cap).
WIDTH, HEIGHT = 1344, 768
# length must sit on the 17k+5 grid: 124 frames @ 24fps = ~5.17s
LENGTH = 124

# lightx2v's distilled LoRAs, each with its own step count and shift contract.
# The 768p v1.0 checkpoint is trained on this exact 1344x768 grid; the 8-step
# one is trained on 544p mixed, so it upscales into 768p off-distribution.
TURBO_LORAS = {
    4: ("minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors", 6.0, 3.0),
    8: ("minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors", 12.0, 3.0),
}

TURBO = "--base" not in sys.argv


def opt(flag, default):
    return float(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


if TURBO:
    nfe = 8 if "--8step" in sys.argv else 4
    TURBO_LORA, SHIFT_VIDEO, SHIFT_AUDIO = TURBO_LORAS[nfe]
    STEPS = nfe
else:
    TURBO_LORA = None
    STEPS, SHIFT_VIDEO, SHIFT_AUDIO = 30, 12.0, 3.0

STEPS = int(opt("--steps", STEPS))
SHIFT_VIDEO = opt("--shift-video", SHIFT_VIDEO)
SHIFT_AUDIO = opt("--shift-audio", SHIFT_AUDIO)

# Once steps drop to 4, weight staging dominates the run, not sampling -- so the
# quant pair's size is the lever. w4a8_mixed is 12.5GB + 16.5GB against
# int8_convrot's 21GB + 27GB.
QUANT = "w4a8" if "--w4a8" in sys.argv else "int8"
UNET_NAME, CLIP_NAME = {
    "int8": (
        "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
    ),
    "w4a8": (
        "minimax_h3_fl2va_pruned_w4a8_mixed.safetensors",
        "qwen3vl_32b_minimax_h3_w4a8_mixed.safetensors",
    ),
}[QUANT]

PROMPT = (
    "A lone street musician plays a saxophone under a flickering neon sign on a "
    "rain-slicked city street at night. Slow dolly-in, shallow depth of field, "
    "cinematic film grain. Audio: a warm melancholic saxophone melody, distant "
    "traffic hum, light rain pattering on pavement."
)

# Same prompt + same seed hits ComfyUI's output cache and returns in ~0s, which
# silently invalidates any timing comparison -- vary one of these when measuring.
if "--prompt" in sys.argv:
    PROMPT = sys.argv[sys.argv.index("--prompt") + 1]
SEED = int(opt("--seed", 42))

workflow = {
    "unet": {
        "class_type": "UNETLoader",
        "inputs": {
            "unet_name": UNET_NAME,
            "weight_dtype": "default",
        },
    },
    "clip": {
        "class_type": "CLIPLoader",
        "inputs": {"clip_name": CLIP_NAME, "type": "minimax"},
    },
    "vae": {
        "class_type": "VAELoader",
        "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"},
    },
    "audio_vae": {
        "class_type": "VAELoader",
        "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"},
    },
    # t2va: no keyframes connected, so this is pure text-to-audio-video
    "cond": {
        "class_type": "MiniMaxH3ImageToVideo",
        "inputs": {
            "clip": ["clip", 0],
            "vae": ["vae", 0],
            "prompt": PROMPT,
            "width": WIDTH,
            "height": HEIGHT,
            "length": LENGTH,
        },
    },
    # shift_video drives the sampler schedule; the DiT derives audio's from it
    "shift": {
        "class_type": "MiniMaxH3SigmaShift",
        "inputs": {
            "model": ["lora" if TURBO else "unet", 0],
            "shift_video": SHIFT_VIDEO,
            "shift_audio": SHIFT_AUDIO,
        },
    },
    "guider": {
        "class_type": "BasicGuider",
        "inputs": {"model": ["shift", 0], "conditioning": ["cond", 0]},
    },
    "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED}},
    "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
    "sigmas": {
        "class_type": "BasicScheduler",
        "inputs": {
            "model": ["shift", 0],
            "scheduler": "simple",
            "steps": STEPS,
            "denoise": 1.0,
        },
    },
    "sample": {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {
            "noise": ["noise", 0],
            "guider": ["guider", 0],
            "sampler": ["sampler", 0],
            "sigmas": ["sigmas", 0],
            "latent_image": ["cond", 1],
        },
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
        "inputs": {
            "images": ["decode_video", 0],
            "fps": 24.0,
            "audio": ["decode_audio", 0],
        },
    },
    "save": {
        "class_type": "SaveVideo",
        "inputs": {
            "video": ["video", 0],
            "filename_prefix": (
                f"video/MiniMax_H3_{'turbo' if TURBO else 'base'}_{QUANT}"
                f"_{STEPS}step_sv{SHIFT_VIDEO:g}_sa{SHIFT_AUDIO:g}"
            ),
            "format": "auto",
            "codec": "auto",
        },
    },
}

if TURBO:
    # patched onto the int8_convrot weights on the fly, so the matmuls stay
    # quantized and only the patched rows get requantized per step
    workflow["lora"] = {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {
            "model": ["unet", 0],
            "lora_name": TURBO_LORA,
            "strength_model": 1.0,
        },
    }


def post(path, payload):
    req = urllib.request.Request(
        SERVER + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def get(path):
    return json.loads(urllib.request.urlopen(SERVER + path, timeout=60).read())


def main():
    print(
        f"submitting t2va [{QUANT} / {TURBO_LORA or 'base'}]: {WIDTH}x{HEIGHT}, {LENGTH} frames "
        f"(~{LENGTH/24:.1f}s), {STEPS} steps, shift {SHIFT_VIDEO}/{SHIFT_AUDIO}"
    )
    try:
        res = post("/prompt", {"prompt": workflow})
    except urllib.error.HTTPError as e:
        print("SUBMIT FAILED:", e.read().decode()[:4000])
        return 1

    pid = res["prompt_id"]
    print("prompt_id:", pid)

    start = time.time()
    while True:
        time.sleep(10)
        hist = get(f"/history/{pid}")
        if pid in hist:
            entry = hist[pid]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                print("FAILED after %.0fs" % (time.time() - start))
                for m in status.get("messages", []):
                    print(" ", json.dumps(m)[:2000])
                return 1
            print("DONE in %.0fs" % (time.time() - start))
            for node_id, out in entry.get("outputs", {}).items():
                for vid in out.get("images", []) + out.get("video", []):
                    print("  output:", vid)
            return 0
        q = get("/queue")
        running = len(q.get("queue_running", []))
        print("  ... %.0fs elapsed (running=%d)" % (time.time() - start, running))


if __name__ == "__main__":
    sys.exit(main())

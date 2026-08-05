#!/usr/bin/env python3
"""MiniMax H3 t2va smoke test: text -> 5s 768p video with native stereo audio.

Runs the int8_convrot quant pair (transformer 21GB + Qwen3-VL-32B TE 27GB) on a
single 48GB L40S, which only fits because ComfyUI swaps the TE out after
encoding. Submits via the /prompt API and polls until the video lands.
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
STEPS = 30

PROMPT = (
    "A lone street musician plays a saxophone under a flickering neon sign on a "
    "rain-slicked city street at night. Slow dolly-in, shallow depth of field, "
    "cinematic film grain. Audio: a warm melancholic saxophone melody, distant "
    "traffic hum, light rain pattering on pavement."
)

workflow = {
    "unet": {
        "class_type": "UNETLoader",
        "inputs": {
            "unet_name": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
            "weight_dtype": "default",
        },
    },
    "clip": {
        "class_type": "CLIPLoader",
        "inputs": {
            "clip_name": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
            "type": "minimax",
        },
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
        "inputs": {"model": ["unet", 0], "shift_video": 12.0, "shift_audio": 3.0},
    },
    "guider": {
        "class_type": "BasicGuider",
        "inputs": {"model": ["shift", 0], "conditioning": ["cond", 0]},
    },
    "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": 42}},
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
            "filename_prefix": "video/MiniMax_H3_smoke",
            "format": "auto",
            "codec": "auto",
        },
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
    print(f"submitting t2va: {WIDTH}x{HEIGHT}, {LENGTH} frames (~{LENGTH/24:.1f}s), {STEPS} steps")
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

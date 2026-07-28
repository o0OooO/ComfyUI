#!/usr/bin/env python3
"""
multiref_compare.py — 同一组参考图 + 同一条提示词,横向对比三个模型的"多参考图生图"效果。

  qwen       Qwen-Image-Edit-2511   最多 3 张参考图  Apache-2.0   20B fp8mixed
  flux       FLUX.2-dev             多张(链式)       权重非商用    32B fp8mixed
  sensenova  SenseNova-U1-8B-MoT    最多 6 张参考图  本地已有      8B

三者机制不同,脚本已各自适配:
  - qwen: 参考图喂给 TextEncodeQwenImageEditPlus 的 image1/2/3(节点只有 3 个槽位),
          图会同时过 Qwen2.5-VL(缩到 384²做视觉理解) 和 VAE(缩到 1024²做 ref_latent)。
  - flux: 参考图各自 VAEEncode 后,用 ReferenceLatent 链式串联 —— 节点自述
          "chain multiple to set multiple reference images",所以张数不受节点槽位限制。
  - sensenova: 走 SenseNovaU1LocalCompose(image + image2..image6),
          prompt 里用 <image> 占位符按序绑定,详见 sensenova_api.py compose。

依赖:仅标准库。需要 ComfyUI 正在运行。
      qwen/flux 权重由 restore_multiref_models.sh 准备。

示例:
  # 三个模型全跑,同一组图 + 同一条 prompt
  python multiref_compare.py --ref charA.png --ref charB.png \
      --prompt "the two people shaking hands in a modern office" \
      --outdir ./cmp

  # 只跑 qwen 和 flux,固定 seed 便于复现
  python multiref_compare.py -m qwen -m flux --ref a.png --ref b.png --ref c.png \
      --prompt "..." --seed 123 --steps 30

  # sensenova 的 prompt 需要 <image> 占位符,可用 --sensenova-prompt 单独给
  python multiref_compare.py --ref charA.png --ref prop.png \
      --prompt "a person holding the product in a bright studio" \
      --sensenova-prompt "<image> 手持 <image>,明亮影棚打光" \
      --outdir ./cmp
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sensenova_api import ComfyClient, DEFAULT_SERVER  # noqa: E402  复用同目录的客户端

# ---------------------------------------------------------------------------
# 权重文件名(与 restore_multiref_models.sh 软链出来的名字一致)
# ---------------------------------------------------------------------------
QWEN_UNET = "qwen_image_edit_2511_fp8mixed.safetensors"
QWEN_CLIP = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
QWEN_VAE = "qwen_image_vae.safetensors"

FLUX_UNET = "flux2_dev_fp8mixed.safetensors"
FLUX_CLIP = "mistral_3_small_flux2_fp8.safetensors"
FLUX_VAE = "flux2-vae.safetensors"
FLUX_TURBO_LORA = "Flux2TurboComfyv2.safetensors"

SENSENOVA_MODEL = "sensenova/SenseNova-U1-8B-MoT"

QWEN_MAX_REFS = 3  # TextEncodeQwenImageEditPlus 只有 image1/2/3
SENSENOVA_MAX_REFS = 6  # SenseNovaU1LocalCompose: image + image2..image6


# ---------------------------------------------------------------------------
# 各模型的 ComfyUI API prompt 构建
# ---------------------------------------------------------------------------
def build_qwen(args, ref_names: list[str]) -> dict:
    """Qwen-Image-Edit-2511:参考图进 TextEncodeQwenImageEditPlus 的 image1/2/3。"""
    refs = ref_names[:QWEN_MAX_REFS]
    n = {
        "unet": {"class_type": "UNETLoader", "inputs": {
            "unet_name": QWEN_UNET, "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": QWEN_CLIP, "type": "qwen_image"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": QWEN_VAE}},
    }
    # 每张参考图一个 LoadImage,接到编码节点的 image1/2/3
    enc_pos = {"clip": ["clip", 0], "prompt": args.prompt, "vae": ["vae", 0]}
    for i, name in enumerate(refs):
        nid = f"img{i}"
        n[nid] = {"class_type": "LoadImage", "inputs": {"image": name}}
        enc_pos[f"image{i + 1}"] = [nid, 0]
    n["pos"] = {"class_type": "TextEncodeQwenImageEditPlus", "inputs": enc_pos}
    n["neg"] = {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {
        "clip": ["clip", 0], "prompt": args.negative, "vae": ["vae", 0]}}

    # 输出画布:第一张参考图缩放到目标边长,保证长宽是 16 的倍数
    n["latent"] = {"class_type": "EmptySD3LatentImage", "inputs": {
        "width": args.width, "height": args.height, "batch_size": 1}}
    n["ksampler"] = {"class_type": "KSampler", "inputs": {
        "model": ["unet", 0], "positive": ["pos", 0], "negative": ["neg", 0],
        "latent_image": ["latent", 0], "seed": args.seed, "steps": args.steps,
        "cfg": args.cfg, "sampler_name": "euler", "scheduler": "simple",
        "denoise": 1.0}}
    n["decode"] = {"class_type": "VAEDecode", "inputs": {
        "samples": ["ksampler", 0], "vae": ["vae", 0]}}
    n["save"] = {"class_type": "SaveImage", "inputs": {
        "images": ["decode", 0], "filename_prefix": "cmp_qwen2511"}}
    return n


def build_flux(args, ref_names: list[str]) -> dict:
    """FLUX.2-dev:每张参考图 VAEEncode 后用 ReferenceLatent 链式串联。

    ReferenceLatent 节点自述可以 chain multiple 来设置多张参考图,
    所以这里把 conditioning 依次穿过 N 个 ReferenceLatent。
    """
    n = {
        "unet": {"class_type": "UNETLoader", "inputs": {
            "unet_name": FLUX_UNET, "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": FLUX_CLIP, "type": "flux2"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": FLUX_VAE}},
    }
    model_src = ["unet", 0]
    if args.flux_turbo:
        # Turbo LoRA:少步数出图,调 prompt 阶段省时间
        n["lora"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["unet", 0], "lora_name": FLUX_TURBO_LORA,
            "strength_model": 1.0}}
        model_src = ["lora", 0]

    n["pos"] = {"class_type": "CLIPTextEncode", "inputs": {
        "clip": ["clip", 0], "text": args.prompt}}

    # 链式 ReferenceLatent:cond -> ref1 -> ref2 -> ... -> refN
    cond = ["pos", 0]
    for i, name in enumerate(ref_names):
        n[f"img{i}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
        n[f"enc{i}"] = {"class_type": "VAEEncode", "inputs": {
            "pixels": [f"img{i}", 0], "vae": ["vae", 0]}}
        n[f"ref{i}"] = {"class_type": "ReferenceLatent", "inputs": {
            "conditioning": cond, "latent": [f"enc{i}", 0]}}
        cond = [f"ref{i}", 0]

    n["guidance"] = {"class_type": "FluxGuidance", "inputs": {
        "conditioning": cond, "guidance": args.flux_guidance}}
    n["neg"] = {"class_type": "CLIPTextEncode", "inputs": {
        "clip": ["clip", 0], "text": args.negative}}

    # Flux2 专用:128 通道 latent + 按 seq_len 计算的 sigmas
    n["latent"] = {"class_type": "EmptyFlux2LatentImage", "inputs": {
        "width": args.width, "height": args.height, "batch_size": 1}}
    n["sigmas"] = {"class_type": "Flux2Scheduler", "inputs": {
        "steps": args.steps, "width": args.width, "height": args.height}}
    n["sampler"] = {"class_type": "KSamplerSelect", "inputs": {
        "sampler_name": "euler"}}
    n["noise"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": args.seed}}
    n["guider"] = {"class_type": "CFGGuider", "inputs": {
        "model": model_src, "positive": ["guidance", 0], "negative": ["neg", 0],
        "cfg": args.flux_cfg}}
    n["adv"] = {"class_type": "SamplerCustomAdvanced", "inputs": {
        "noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0],
        "sigmas": ["sigmas", 0], "latent_image": ["latent", 0]}}
    n["decode"] = {"class_type": "VAEDecode", "inputs": {
        "samples": ["adv", 0], "vae": ["vae", 0]}}
    n["save"] = {"class_type": "SaveImage", "inputs": {
        "images": ["decode", 0], "filename_prefix": "cmp_flux2dev"}}
    return n


def build_sensenova(args, ref_names: list[str]) -> dict:
    """SenseNova-U1 Compose:image + image2..image6,prompt 用 <image> 占位符绑定。"""
    refs = ref_names[:SENSENOVA_MAX_REFS]
    prompt = args.sensenova_prompt or args.prompt
    # 输入项与 sensenova_api.py 的 loader_node 保持一致(已按节点 schema 核对)
    n = {"loader": {"class_type": "SenseNovaU1LocalLoader", "inputs": {
        "model_path": SENSENOVA_MODEL, "sensenova_u1_src": "", "device": "cuda",
        "dtype": "bfloat16", "attn_backend": "auto", "device_map": "none",
        "max_memory": "", "vram_mode": args.vram_mode, "gguf_checkpoint": ""}}}
    slots = ["image", "image2", "image3", "image4", "image5", "image6"]
    ci = {
        "u1_model": ["loader", 0], "prompt": prompt,
        "auto_size": False, "width": args.width, "height": args.height,
        "target_megapixels": 1.048576, "input_megapixels": args.input_mp,
        "cfg_scale": args.cfg, "img_cfg_scale": 1.0, "cfg_norm": "none",
        "timestep_shift": 3.0, "cfg_interval_start": 0.0, "cfg_interval_end": 1.0,
        "num_steps": args.steps, "seed": args.seed, "think_mode": False,
    }
    for i, name in enumerate(refs):
        nid = f"img{i}"
        n[nid] = {"class_type": "LoadImage", "inputs": {"image": name}}
        ci[slots[i]] = [nid, 0]
    n["compose"] = {"class_type": "SenseNovaU1LocalCompose", "inputs": ci}
    n["save"] = {"class_type": "SaveImage", "inputs": {
        "images": ["compose", 0], "filename_prefix": "cmp_sensenova"}}
    return n


BUILDERS = {"qwen": build_qwen, "flux": build_flux, "sensenova": build_sensenova}
LABELS = {
    "qwen": "Qwen-Image-Edit-2511 (20B, Apache-2.0)",
    "flux": "FLUX.2-dev (32B, 权重非商用)",
    "sensenova": "SenseNova-U1-8B-MoT (8B, 本地)",
}


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="多参考图生图:Qwen-Image-Edit-2511 / FLUX.2-dev / SenseNova-U1 横向对比",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("示例:")[-1])
    p.add_argument("-m", "--model", action="append", choices=list(BUILDERS),
                   help="要跑的模型,可多次传入(默认三个都跑)")
    p.add_argument("--ref", action="append", required=True,
                   help="参考图路径,可多次传入(qwen 只用前 3 张,sensenova 前 6 张)")
    p.add_argument("--prompt", required=True, help="提示词(qwen/flux 用)")
    p.add_argument("--sensenova-prompt", default="", dest="sensenova_prompt",
                   help="SenseNova 专用提示词(需 <image> 占位符按序绑定;不给则复用 --prompt)")
    p.add_argument("--negative", default="", help="负向提示词(sensenova 不支持,忽略)")
    p.add_argument("--outdir", default="./multiref_cmp", help="输出目录")
    p.add_argument("--width", type=int, default=1024, help="输出宽(16 的倍数)")
    p.add_argument("--height", type=int, default=1024, help="输出高(16 的倍数)")
    p.add_argument("--steps", type=int, default=30, help="采样步数")
    p.add_argument("--seed", type=int, default=42, help="随机种子(三个模型共用,便于复现)")
    p.add_argument("--cfg", type=float, default=4.0, help="CFG(qwen/sensenova)")
    p.add_argument("--flux-cfg", type=float, default=1.0, dest="flux_cfg",
                   help="FLUX.2 的 CFG(用 FluxGuidance 时通常保持 1.0)")
    p.add_argument("--flux-guidance", type=float, default=4.0, dest="flux_guidance",
                   help="FluxGuidance 强度")
    p.add_argument("--flux-turbo", action="store_true", dest="flux_turbo",
                   help="FLUX.2 挂 Turbo LoRA(少步数快出图,适合调 prompt)")
    p.add_argument("--input-mp", type=float, default=1.048576, dest="input_mp",
                   help="SenseNova 每张输入图像素上限 MP(多图易 OOM 时调小)")
    p.add_argument("--vram-mode", default="full", dest="vram_mode",
                   choices=["full", "low", "balanced"],
                   help="SenseNova 显存模式(与 sensenova_api.py 一致)")
    p.add_argument("--server", default=DEFAULT_SERVER, help="ComfyUI 地址")
    return p.parse_args()


def main():
    args = parse_args()
    models = args.model or ["qwen", "flux", "sensenova"]

    for w, label in ((args.width, "width"), (args.height, "height")):
        if w % 16:
            sys.exit(f"[错误] --{label} 需为 16 的倍数,收到 {w}")

    client = ComfyClient(args.server)
    client.ping()

    os.makedirs(args.outdir, exist_ok=True)

    # 参考图只上传一次,三个模型共用同一批服务端文件名 —— 保证对比公平
    print(f"[上传] {len(args.ref)} 张参考图")
    ref_names = [client.upload_image(p) for p in args.ref]
    for p, n in zip(args.ref, ref_names):
        print(f"        {os.path.basename(p)} -> {n}")

    if len(args.ref) > QWEN_MAX_REFS and "qwen" in models:
        print(f"[注意] 传了 {len(args.ref)} 张,Qwen 节点只有 3 个图槽位,"
              f"只用前 {QWEN_MAX_REFS} 张")
    if len(args.ref) > SENSENOVA_MAX_REFS and "sensenova" in models:
        print(f"[注意] SenseNova 最多 6 张,只用前 {SENSENOVA_MAX_REFS} 张")
    if "sensenova" in models and not args.sensenova_prompt \
            and "<image>" not in args.prompt:
        print("[注意] SenseNova compose 通常需要 prompt 里带 <image> 占位符按序绑定参考图;"
              "当前 prompt 没有占位符,效果可能不如预期 —— 可用 --sensenova-prompt 单独指定")

    results = {}
    for m in models:
        print(f"\n{'=' * 60}\n{LABELS[m]}\n{'=' * 60}")
        prompt_graph = BUILDERS[m](args, ref_names)
        t0 = time.time()
        try:
            pid = client.submit(prompt_graph)
            print(f"[提交] prompt_id={pid},等待中(首次加载权重较慢)...")
            outputs = client.wait(pid)
            out = os.path.join(args.outdir, f"{m}.png")
            saved = client.download_images(outputs, out)
            dt = time.time() - t0
            results[m] = {"ok": True, "sec": round(dt, 1), "files": saved}
            print(f"[完成] {dt:.1f}s -> {', '.join(saved)}")
        except SystemExit as e:
            # ComfyClient 用 sys.exit 报错;这里接住,让其余模型继续跑
            results[m] = {"ok": False, "sec": round(time.time() - t0, 1),
                          "error": str(e)}
            print(f"[失败] {e}")

    print(f"\n{'=' * 60}\n对比汇总\n{'=' * 60}")
    for m in models:
        r = results.get(m, {})
        status = f"OK   {r.get('sec')}s" if r.get("ok") else f"FAIL {r.get('error', '')[:60]}"
        print(f"  {m:<10} {status}")
    meta = os.path.join(args.outdir, "compare_meta.json")
    with open(meta, "w") as f:
        json.dump({"prompt": args.prompt, "sensenova_prompt": args.sensenova_prompt,
                   "refs": args.ref, "seed": args.seed, "steps": args.steps,
                   "size": [args.width, args.height], "results": results},
                  f, ensure_ascii=False, indent=2)
    print(f"\n输出目录:{args.outdir}\n参数记录:{meta}")


if __name__ == "__main__":
    main()

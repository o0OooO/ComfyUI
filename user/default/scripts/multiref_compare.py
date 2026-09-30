#!/usr/bin/env python3
"""
multiref_compare.py — 同一组参考图 + 同一条提示词,横向对比各模型的"多参考图生图"效果。

当代(默认跑这三个):

  qwen21     Qwen-Image-2.1         最多 10 张参考图  Qwen Research(非商用)  20B int8
  flux       FLUX.2-dev             多张(链式)        权重非商用              32B fp8mixed
  u15        SenseNova-U1.5-8B-MoT  最多 10 张参考图  本地已有                8B + 8 步 LoRA

上一代(留作对照,--model 显式指定才跑):

  qwen       Qwen-Image-Edit-2511   最多 3 张参考图   Apache-2.0   20B fp8mixed
  sensenova  SenseNova-U1-8B-MoT    最多 6 张参考图   本地已有      8B

机制各不相同,脚本已各自适配:
  - qwen21: 参考图走 TextEncodeQwenImage21 的 autogrow 槽 images.image_1..image_10,
          提示词里用 <image1>..<image10> 点名;cfg 必须是 1(负向走 negative_prompt)。
  - qwen: 参考图喂给 TextEncodeQwenImageEditPlus 的 image1/2/3(节点只有 3 个槽位),
          图会同时过 Qwen2.5-VL(缩到 384²做视觉理解) 和 VAE(缩到 1024²做 ref_latent)。
  - flux: 参考图各自 VAEEncode 后,用 ReferenceLatent 链式串联 —— 节点自述
          "chain multiple to set multiple reference images",所以张数不受节点槽位限制。
  - u15: 纯核心节点。参考图走共享的 HiDreamO1ReferenceImages(images.image_1..image_100),
          latent 是像素空间 EmptyHiDreamO1LatentImage;默认挂 8 步蒸馏 LoRA(cfg 1.0)。
  - sensenova: 走 SenseNovaU1LocalCompose(image + image2..image6),
          prompt 里用 <image> 占位符按序绑定,详见 sensenova_api.py compose。

依赖:仅标准库。需要 ComfyUI 正在运行。
      qwen/flux 权重由 restore_multiref_models.sh 准备。

示例:
  # 当代三个全跑(qwen21 / flux / u15),同一组图 + 同一条 prompt
  python multiref_compare.py --ref charA.png --ref charB.png \
      --prompt "the two people shaking hands in a modern office" \
      --outdir ./cmp

  # 新老同台:2.1 vs 2511、U1.5 vs U1
  python multiref_compare.py -m qwen21 -m qwen -m u15 -m sensenova \
      --ref a.png --ref b.png --ref c.png --prompt "..." --seed 123 --steps 30

  # qwen21 要用 <imageN> 点名参考图,u15 直接写中文指令
  python multiref_compare.py -m qwen21 -m u15 --ref charA.png --ref prop.png \
      --prompt "把 <image1> 里的人手持 <image2> 的产品,明亮影棚打光" \
      --qwen21-prompt "The person from <image1> holding the product from <image2>, bright studio lighting" \
      --outdir ./cmp

  # sensenova(上一代)的 prompt 需要 <image> 占位符,可用 --sensenova-prompt 单独给
  python multiref_compare.py -m sensenova --ref charA.png --ref prop.png \
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

QWEN21_UNET = "qwen_image_2.1_int8_convrot.safetensors"
QWEN21_CLIP = "qwen3vl_8b_int8_convrot.safetensors"
QWEN21_VAE = "qwen_image_2.1_vae_bf16.safetensors"

U15_CKPT = "SenseNova-U1.5-8B-MoT-T8-int8-convrot-tagged.safetensors"
# 官方 V2 原始文件缺 diffusion_model. 前缀,加载不生效;-comfy 那份是补过前缀的
U15_LORA_8STEP = "SenseNova-U1.5-8B-MoT-LoRA-8step-V2-comfy.safetensors"

QWEN_MAX_REFS = 3  # TextEncodeQwenImageEditPlus 只有 image1/2/3
QWEN21_MAX_REFS = 10  # TextEncodeQwenImage21: <image1>..<image10>
SENSENOVA_MAX_REFS = 6  # SenseNovaU1LocalCompose: image + image2..image6
U15_MAX_REFS = 10  # 节点支持到 image_100,这里按实用上限收在 10


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


def build_qwen21(args, ref_names: list[str]) -> dict:
    """Qwen-Image-2.1:参考图走 autogrow 槽 images.image_1..image_10。

    和 2511 的三个差别:
      - 槽位是 autogrow 的点号键(images.image_N),不是固定的 image1/2/3;
      - resolution 是**总像素预算**(0 = 保持 image_1 原尺寸,只对齐到 32);
      - 负向走 negative_prompt 编进同一次前向,所以 KSampler 的 cfg 必须是 1。
    """
    refs = ref_names[:QWEN21_MAX_REFS]
    prompt = args.qwen21_prompt or args.prompt
    n = {
        "unet": {"class_type": "UNETLoader", "inputs": {
            "unet_name": QWEN21_UNET, "weight_dtype": "default"}},
        "cache": {"class_type": "QwenImage21Cache", "inputs": {
            "model": ["unet", 0], "device": "auto", "dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": QWEN21_CLIP, "type": "qwen_image", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": QWEN21_VAE}},
    }
    enc = {"clip": ["clip", 0], "vae": ["vae", 0], "prompt": prompt,
           "negative_prompt": args.negative, "resolution": args.width}
    for i, name in enumerate(refs):
        nid = f"img{i}"
        n[nid] = {"class_type": "LoadImage", "inputs": {"image": name}}
        enc[f"images.image_{i + 1}"] = [nid, 0]
    n["enc"] = {"class_type": "TextEncodeQwenImage21", "inputs": enc}

    # 固定画布,和其他模型同尺寸便于对比。想让画布跟随 image_1 就把 latent 换成 ["enc", 2]
    n["latent"] = {"class_type": "EmptyLatentImage", "inputs": {
        "width": args.width, "height": args.height, "batch_size": 1}}
    n["ksampler"] = {"class_type": "KSampler", "inputs": {
        "model": ["cache", 0], "positive": ["enc", 0], "negative": ["enc", 1],
        "latent_image": ["latent", 0], "seed": args.seed, "steps": args.steps,
        "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}}
    n["decode"] = {"class_type": "VAEDecode", "inputs": {
        "samples": ["ksampler", 0], "vae": ["vae", 0]}}
    n["save"] = {"class_type": "SaveImage", "inputs": {
        "images": ["decode", 0], "filename_prefix": "cmp_qwen2.1"}}
    return n


def build_u15(args, ref_names: list[str]) -> dict:
    """SenseNova-U1.5-8B-MoT:纯核心节点,不需要自定义节点也不走 wrapper 推理脚本。

    一个 ckpt 同时出 MODEL/CLIP/VAE(TE 和 VAE 是核心合成的哨兵);
    latent 是像素空间([B,3,H,W]),所以画布用 EmptyHiDreamO1LatentImage;
    参考图走共享的 HiDreamO1ReferenceImages(images.image_1..image_100)。
    """
    refs = ref_names[:U15_MAX_REFS]
    n = {"ckpt": {"class_type": "CheckpointLoaderSimple",
                  "inputs": {"ckpt_name": U15_CKPT}}}
    model_src = ["ckpt", 0]
    steps, cfg = args.steps, args.cfg
    if not args.u15_base:
        n["lora"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["ckpt", 0], "lora_name": U15_LORA_8STEP,
            "strength_model": 1.0}}
        model_src = ["lora", 0]
        steps, cfg = 8, 1.0  # 官方 8 步配方,--steps/--cfg 在这一档不生效
    # 官方 --timestep_shift 3.0;resolution_noise_scale 是自动的,不用额外节点
    n["shift"] = {"class_type": "SenseNovaSamplingOptions", "inputs": {
        "model": model_src, "shift": 3.0}}

    n["pos"] = {"class_type": "CLIPTextEncode", "inputs": {
        "clip": ["ckpt", 1], "text": args.prompt}}
    n["neg"] = {"class_type": "CLIPTextEncode", "inputs": {
        "clip": ["ckpt", 1], "text": args.negative}}

    pos_src, neg_src = ["pos", 0], ["neg", 0]
    if refs:
        ri = {"positive": ["pos", 0], "negative": ["neg", 0]}
        for i, name in enumerate(refs):
            nid = f"img{i}"
            n[nid] = {"class_type": "LoadImage", "inputs": {"image": name}}
            ri[f"images.image_{i + 1}"] = [nid, 0]
        n["refs"] = {"class_type": "HiDreamO1ReferenceImages", "inputs": ri}
        pos_src, neg_src = ["refs", 0], ["refs", 1]

    n["latent"] = {"class_type": "EmptyHiDreamO1LatentImage", "inputs": {
        "width": args.width, "height": args.height, "batch_size": 1}}
    n["sampler"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
    # normal + SenseNovaModelSampling = 官方 upstream_sigmas,别换 scheduler
    n["sigmas"] = {"class_type": "BasicScheduler", "inputs": {
        "model": ["shift", 0], "scheduler": "normal", "steps": steps, "denoise": 1.0}}
    n["ks"] = {"class_type": "SamplerCustom", "inputs": {
        "model": ["shift", 0], "add_noise": True, "noise_seed": args.seed, "cfg": cfg,
        "positive": pos_src, "negative": neg_src,
        "sampler": ["sampler", 0], "sigmas": ["sigmas", 0],
        "latent_image": ["latent", 0]}}
    n["decode"] = {"class_type": "VAEDecode", "inputs": {
        "samples": ["ks", 0], "vae": ["ckpt", 2]}}
    n["save"] = {"class_type": "SaveImage", "inputs": {
        "images": ["decode", 0], "filename_prefix": "cmp_u1.5"}}
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


BUILDERS = {"qwen21": build_qwen21, "flux": build_flux, "u15": build_u15,
            "qwen": build_qwen, "sensenova": build_sensenova}
LABELS = {
    "qwen21": "Qwen-Image-2.1 (20B int8, Qwen Research 非商用)",
    "flux": "FLUX.2-dev (32B, 权重非商用)",
    "u15": "SenseNova-U1.5-8B-MoT (8B, 本地)",
    "qwen": "Qwen-Image-Edit-2511 (20B, Apache-2.0) [上一代]",
    "sensenova": "SenseNova-U1-8B-MoT (8B, 本地) [上一代]",
}
DEFAULT_MODELS = ["qwen21", "flux", "u15"]
MAX_REFS = {"qwen21": QWEN21_MAX_REFS, "qwen": QWEN_MAX_REFS,
            "u15": U15_MAX_REFS, "sensenova": SENSENOVA_MAX_REFS}


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="多参考图生图:Qwen-Image-Edit-2511 / FLUX.2-dev / SenseNova-U1 横向对比",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("示例:")[-1])
    p.add_argument("-m", "--model", action="append", choices=list(BUILDERS),
                   help=f"要跑的模型,可多次传入(默认 {' '.join(DEFAULT_MODELS)})")
    p.add_argument("--ref", action="append", required=True,
                   help="参考图路径,可多次传入(超出各模型上限的会被截掉,见 --help 顶部说明)")
    p.add_argument("--prompt", required=True, help="提示词(qwen21/qwen/flux/u15 用)")
    p.add_argument("--qwen21-prompt", default="", dest="qwen21_prompt",
                   help="Qwen-Image-2.1 专用提示词(用 <image1>..<image10> 点名参考图;不给则复用 --prompt)")
    p.add_argument("--u15-base", action="store_true", dest="u15_base",
                   help="U1.5 走无 LoRA 的基础档(用 --steps/--cfg);默认是 8 步蒸馏档 cfg 1.0")
    p.add_argument("--sensenova-prompt", default="", dest="sensenova_prompt",
                   help="SenseNova-U1 专用提示词(需 <image> 占位符按序绑定;不给则复用 --prompt)")
    p.add_argument("--negative", default="", help="负向提示词(sensenova 不支持,忽略)")
    p.add_argument("--outdir", default="./multiref_cmp", help="输出目录")
    p.add_argument("--width", type=int, default=1024, help="输出宽(32 的倍数)")
    p.add_argument("--height", type=int, default=1024, help="输出高(32 的倍数)")
    p.add_argument("--steps", type=int, default=30, help="采样步数(u15 蒸馏档固定 8 步)")
    p.add_argument("--seed", type=int, default=42, help="随机种子(所有模型共用,便于复现)")
    p.add_argument("--cfg", type=float, default=4.0,
                   help="CFG(qwen/sensenova/u15 基础档;qwen21 强制 1,u15 蒸馏档强制 1.0)")
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
    models = args.model or list(DEFAULT_MODELS)

    # 32 是最严的那档(qwen21 对齐 32、u15 的像素空间 latent 也要 32),统一按 32 卡
    for w, label in ((args.width, "width"), (args.height, "height")):
        if w % 32:
            sys.exit(f"[错误] --{label} 需为 32 的倍数,收到 {w}")

    client = ComfyClient(args.server)
    client.ping()

    os.makedirs(args.outdir, exist_ok=True)

    # 参考图只上传一次,三个模型共用同一批服务端文件名 —— 保证对比公平
    print(f"[上传] {len(args.ref)} 张参考图")
    ref_names = [client.upload_image(p) for p in args.ref]
    for p, n in zip(args.ref, ref_names):
        print(f"        {os.path.basename(p)} -> {n}")

    for m in models:
        cap = MAX_REFS.get(m)
        if cap and len(args.ref) > cap:
            print(f"[注意] 传了 {len(args.ref)} 张,{m} 上限 {cap} 张,只用前 {cap} 张")
    if "qwen21" in models:
        tag_prompt = args.qwen21_prompt or args.prompt
        if not any(f"<image{i}>" in tag_prompt for i in range(1, QWEN21_MAX_REFS + 1)):
            print("[注意] Qwen-Image-2.1 要在提示词里用 <image1>..<imageN> 点名参考图,"
                  "不点名的容易被当背景信息忽略 —— 可用 --qwen21-prompt 单独指定")
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

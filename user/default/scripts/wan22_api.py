#!/usr/bin/env python3
"""
wan22_api.py — 通过 ComfyUI HTTP API 驱动 Wan2.2 完成各类视频生成任务的统一 CLI。

一个脚本按 --task 参数切换场景,覆盖 Wan2.2 支持的各类视频生成能力:

  t2v        文本生视频           （需 wan2.2_t2v 权重）
  i2v        图生视频             （需 wan2.2_i2v high+low 权重 —— 本环境已具备）
  flf2v      首尾帧生视频          （需 wan2.2_i2v high+low 权重）
  fun_inpaint 首尾帧补全(Fun)      （需 wan2.2_fun_inpaint high+low 权重）
  fun_control 控制视频驱动(Fun)    （需 wan2.2_fun_control high+low 权重 + 控制视频）
  vace       多参考 / 控制视频生视频（需 wan2.1_vace 权重 —— 多参考主力）
  phantom    多主体参考生视频       （需 wan phantom 权重）
  s2v        音频驱动 / 人物口型     （需 wan2.2_s2v 权重 + audio encoder + 音频）
  animate    动作驱动 / 角色动画     （需 wan animate 权重 + 姿态/人脸视频）

⚠️ 重要:不同 --task 需要不同的模型权重。本环境当前只下了 i2v 的权重,
   其它 task 需先把对应权重放进 models/diffusion_models/(及 vae/text_encoders/
   audio_encoders)。每个 task 缺哪些权重,见 --list-models 输出或下方常量。

公共管线(Wan2.2 14B 双模型架构):
  CLIPLoader(umt5) -> CLIPTextEncode x2(正/负) -> VAELoader(wan2.1)
  -> [功能节点 *ToVideo] -> ModelSamplingSD3 -> KSamplerAdvanced(high noise, 前半段)
  -> KSamplerAdvanced(low noise, 后半段) -> VAEDecode -> CreateVideo -> SaveVideo

依赖:仅标准库。需要一个正在运行、装好 Wan 节点与权重的 ComfyUI 服务。

示例:
  # 图生视频(本环境可直接跑)
  python wan22_api.py i2v --image cat.png --prompt "the cat starts running" -o out.mp4

  # 文本生视频
  python wan22_api.py t2v --prompt "a dragon flying over mountains" -o out.mp4

  # 首尾帧
  python wan22_api.py flf2v --start first.png --end last.png --prompt "morphing" -o out.mp4

  # 多参考(VACE):用多张参考图 + 可选控制视频
  python wan22_api.py vace --ref ref1.png --prompt "..." -o out.mp4
  python wan22_api.py vace --ref ref1.png --control-video drive.mp4 --prompt "..." -o out.mp4

  # 多主体参考(Phantom):多张主体图合成进同一视频
  python wan22_api.py phantom --ref subj1.png --ref subj2.png --prompt "..." -o out.mp4

  # 音频驱动口型(S2V)
  python wan22_api.py s2v --image portrait.png --audio speech.wav --prompt "talking" -o out.mp4

  # 动作驱动(Animate)
  python wan22_api.py animate --image char.png --pose-video dance.mp4 --prompt "..." -o out.mp4

  # 查看每个 task 需要的权重
  python wan22_api.py --list-models
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
import urllib.request
import urllib.parse
import urllib.error

DEFAULT_SERVER = os.environ.get("COMFYUI_SERVER", "http://127.0.0.1:8188")

# ----------------------------------------------------------------------------
# 默认模型文件名(取自官方 Wan2.2 工作流模板,可用 --* 覆盖)
# ----------------------------------------------------------------------------
CLIP_NAME = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
VAE_NAME = "wan_2.1_vae.safetensors"

# 各 task 需要的 UNet 权重(high, low);单模型的 low 留空
# lightx2v 4步加速 lora(高/低噪),官方模板用它把步数降到 4
LORA_HIGH = "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors"
LORA_LOW = "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"

# task -> 默认权重配置。weights:(high_unet, low_unet);extra:额外所需文件说明
TASK_MODELS = {
    "t2v":        {"high": "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
                   "low":  "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"},
    "i2v":        {"high": "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
                   "low":  "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"},
    "flf2v":      {"high": "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
                   "low":  "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"},
    "fun_inpaint":{"high": "wan2.2_fun_inpaint_high_noise_14B_fp8_scaled.safetensors",
                   "low":  "wan2.2_fun_inpaint_low_noise_14B_fp8_scaled.safetensors"},
    "fun_control":{"high": "wan2.2_fun_control_high_noise_14B_fp8_scaled.safetensors",
                   "low":  "wan2.2_fun_control_low_noise_14B_fp8_scaled.safetensors"},
    "vace":       {"high": "wan2.1_vace_14B_fp16.safetensors", "low": "",
                   "extra": "Comfy-Org fp16 单模型;可选 causvid lora 加速"},
    "phantom":    {"high": "wan2.1_phantom_14B_fp8.safetensors", "low": "",
                   "extra": "Kijai fp8(Comfy-Org 无 phantom)"},
    "s2v":        {"high": "wan2.2_s2v_14B_fp8_scaled.safetensors", "low": "",
                   "extra": "audio_encoders/wav2vec2_large_english_fp16.safetensors + 音频文件"},
    "animate":    {"high": "wan2.2_animate_14B_fp8_scaled.safetensors", "low": "",
                   "extra": "Kijai fp8 v2(Comfy-Org 仅 bf16);姿态视频 / 人脸视频"},
}


# ----------------------------------------------------------------------------
# ComfyUI API 客户端(与 sensenova_api.py 同款)
# ----------------------------------------------------------------------------
class ComfyClient:
    def __init__(self, server: str):
        self.server = server.rstrip("/")
        self.client_id = uuid.uuid4().hex

    def _get(self, path: str) -> dict:
        with urllib.request.urlopen(f"{self.server}{path}", timeout=60) as r:
            return json.loads(r.read())

    def ping(self) -> None:
        try:
            self._get("/system_stats")
        except Exception as e:
            sys.exit(f"[错误] 无法连接 ComfyUI 服务 {self.server}:{e}")

    def object_info(self) -> dict:
        try:
            return self._get("/object_info")
        except Exception:
            return {}

    def upload_file(self, filepath: str, kind: str = "image") -> str:
        """上传图片/视频/音频到 ComfyUI input 目录,返回服务端文件名。kind 仅用于报错提示。"""
        if not os.path.isfile(filepath):
            sys.exit(f"[错误] 找不到{kind}文件:{filepath}")
        filename = os.path.basename(filepath)
        boundary = "----wan" + uuid.uuid4().hex
        with open(filepath, "rb") as f:
            content = f.read()
        parts = [
            f"--{boundary}\r\n".encode(),
            (f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
             f"Content-Type: application/octet-stream\r\n\r\n").encode(),
            content, f"\r\n--{boundary}\r\n".encode(),
            'Content-Disposition: form-data; name="overwrite"\r\n\r\n'.encode(),
            b"true", f"\r\n--{boundary}--\r\n".encode(),
        ]
        data = b"".join(parts)
        req = urllib.request.Request(
            f"{self.server}/upload/image", data=data,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        with urllib.request.urlopen(req, timeout=300) as r:
            resp = json.loads(r.read())
        name, sub = resp.get("name", filename), resp.get("subfolder", "")
        return f"{sub}/{name}" if sub else name

    def submit(self, prompt: dict) -> str:
        data = json.dumps({"prompt": prompt, "client_id": self.client_id}).encode()
        req = urllib.request.Request(f"{self.server}/prompt", data=data,
                                     headers={"Content-Type": "application/json"})
        try:
            resp = json.loads(urllib.request.urlopen(req, timeout=60).read())
        except urllib.error.HTTPError as e:
            sys.exit(f"[错误] 提交失败 HTTP {e.code}:\n{e.read().decode()[:2000]}")
        if resp.get("node_errors"):
            sys.exit(f"[错误] 节点校验失败:\n{json.dumps(resp['node_errors'], ensure_ascii=False, indent=2)}")
        return resp["prompt_id"]

    def wait(self, prompt_id: str, poll: float = 3.0, timeout: float = 3600) -> dict:
        start = time.time()
        while True:
            hist = self._get(f"/history/{prompt_id}")
            if prompt_id in hist:
                info = hist[prompt_id]
                st = info.get("status", {})
                if st.get("completed") or st.get("status_str") == "success":
                    return info.get("outputs", {})
                if st.get("status_str") == "error":
                    sys.exit(f"[错误] 执行失败:\n{json.dumps(st.get('messages', []), ensure_ascii=False, indent=2)}")
            if time.time() - start > timeout:
                sys.exit(f"[错误] 等待超时({timeout}s),prompt_id={prompt_id}")
            time.sleep(poll)

    def download(self, outputs: dict, out_path: str) -> list[str]:
        """抓取 SaveVideo / VHS / 图片输出。视频在 outputs 的 'gifs'/'images'/'videos' 字段。"""
        saved, idx = [], 0
        base, ext = os.path.splitext(out_path)
        ext = ext or ".mp4"
        for node_id, out in outputs.items():
            items = []
            for key in ("gifs", "videos", "images"):
                items += out.get(key, [])
            for it in items:
                params = urllib.parse.urlencode({
                    "filename": it["filename"], "subfolder": it.get("subfolder", ""),
                    "type": it.get("type", "output")})
                with urllib.request.urlopen(f"{self.server}/view?{params}", timeout=300) as r:
                    data = r.read()
                if len(data) < 256:
                    continue
                target = out_path if idx == 0 else f"{base}_{idx}{ext}"
                with open(target, "wb") as f:
                    f.write(data)
                saved.append(target)
                idx += 1
        return saved


# ----------------------------------------------------------------------------
# 公共节点片段构建
# ----------------------------------------------------------------------------
NEG_DEFAULT = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
               "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
               "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
               "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")


def base_nodes(args):
    """返回(nodes, ids):公共的 clip/vae/text encode 节点。ids 给出常用引用。"""
    n = {
        "clip": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": args.clip, "type": "wan", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": args.vae}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {
            "clip": ["clip", 0], "text": args.prompt}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {
            "clip": ["clip", 0], "text": args.negative}},
    }
    return n


def add_dual_model(nodes, args, high_unet, low_unet, use_lora=True):
    """加载 high/low 双 UNet(+可选 lightx2v lora)+ ModelSamplingSD3。
    返回(model_high_ref, model_low_ref)供 KSampler 用。单模型时 low 复用 high。"""
    nodes["unet_high"] = {"class_type": "UNETLoader", "inputs": {
        "unet_name": high_unet, "weight_dtype": "default"}}
    mh = ["unet_high", 0]
    if use_lora and args.lora_high:
        nodes["lora_high"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["unet_high", 0], "lora_name": args.lora_high, "strength_model": 1.0}}
        mh = ["lora_high", 0]
    nodes["msd3_high"] = {"class_type": "ModelSamplingSD3", "inputs": {
        "model": mh, "shift": args.shift}}

    if low_unet:
        nodes["unet_low"] = {"class_type": "UNETLoader", "inputs": {
            "unet_name": low_unet, "weight_dtype": "default"}}
        ml = ["unet_low", 0]
        if use_lora and args.lora_low:
            nodes["lora_low"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
                "model": ["unet_low", 0], "lora_name": args.lora_low, "strength_model": 1.0}}
            ml = ["lora_low", 0]
        nodes["msd3_low"] = {"class_type": "ModelSamplingSD3", "inputs": {
            "model": ml, "shift": args.shift}}
        return ["msd3_high", 0], ["msd3_low", 0]
    # 单模型:high 和 low 同一个
    return ["msd3_high", 0], ["msd3_high", 0]


def add_dual_sampler(nodes, model_high, model_low, cond_node, args, latent_slot=2):
    """Wan2.2 双段接力采样:high noise 跑前半 -> low noise 跑后半。
    cond_node:产出 (positive,negative,latent) 的功能节点 id。
    返回最终 latent 的引用。"""
    steps = args.steps
    boundary = max(1, steps // 2)  # high/low 切换点
    # 第一段:high noise,start_step=0 -> end_step=boundary,return_with_leftover_noise=enable
    nodes["ks_high"] = {"class_type": "KSamplerAdvanced", "inputs": {
        "model": model_high, "add_noise": "enable", "noise_seed": args.seed,
        "steps": steps, "cfg": args.cfg, "sampler_name": args.sampler,
        "scheduler": args.scheduler, "start_at_step": 0, "end_at_step": boundary,
        "return_with_leftover_noise": "enable",
        "positive": [cond_node, 0], "negative": [cond_node, 1],
        "latent_image": [cond_node, latent_slot]}}
    # 第二段:low noise,从 boundary 跑到结束,add_noise=disable
    nodes["ks_low"] = {"class_type": "KSamplerAdvanced", "inputs": {
        "model": model_low, "add_noise": "disable", "noise_seed": args.seed,
        "steps": steps, "cfg": args.cfg, "sampler_name": args.sampler,
        "scheduler": args.scheduler, "start_at_step": boundary, "end_at_step": 10000,
        "return_with_leftover_noise": "disable",
        "positive": [cond_node, 0], "negative": [cond_node, 1],
        "latent_image": ["ks_high", 0]}}
    return ["ks_low", 0]


def add_output(nodes, latent_ref, args, audio_ref=None):
    """VAEDecode -> CreateVideo -> SaveVideo。"""
    nodes["decode"] = {"class_type": "VAEDecode", "inputs": {
        "samples": latent_ref, "vae": ["vae", 0]}}
    cv_inputs = {"images": ["decode", 0], "fps": args.fps}
    if audio_ref is not None:
        cv_inputs["audio"] = audio_ref
    nodes["create_video"] = {"class_type": "CreateVideo", "inputs": cv_inputs}
    nodes["save"] = {"class_type": "SaveVideo", "inputs": {
        "video": ["create_video", 0], "filename_prefix": "video/Wan22",
        "format": "auto", "codec": "auto"}}


# ----------------------------------------------------------------------------
# 各场景 prompt 构建
# ----------------------------------------------------------------------------
def _wh(args):
    return args.width, args.height


def _batch_images(nodes, load_ids, prefix="imgbatch"):
    """把多个 LoadImage 节点用原生 ImageBatch 链式合并(image1+image2 两两)。
    返回最终图像 batch 的引用。单张直接返回该图。"""
    if len(load_ids) == 1:
        return [load_ids[0], 0]
    prev = [load_ids[0], 0]
    for i, lid in enumerate(load_ids[1:], start=1):
        bid = f"{prefix}_{i}"
        nodes[bid] = {"class_type": "ImageBatch", "inputs": {
            "image1": prev, "image2": [lid, 0]}}
        prev = [bid, 0]
    return prev


def build_t2v(args, client) -> dict:
    n = base_nodes(args)
    m = TASK_MODELS["t2v"]
    mh, ml = add_dual_model(n, args, args.unet_high or m["high"], args.unet_low or m["low"])
    w, h = _wh(args)
    # t2v 没有图像输入,用 EmptyHunyuanLatentVideo / Wan 的空 latent。
    # 官方 t2v 走 ModelSamplingSD3 + 普通 latent;这里用 WanImageToVideo 不传 start_image 等价空驱动。
    n["cond"] = {"class_type": "WanImageToVideo", "inputs": {
        "positive": ["pos", 0], "negative": ["neg", 0], "vae": ["vae", 0],
        "width": w, "height": h, "length": args.length, "batch_size": 1}}
    latent = add_dual_sampler(n, mh, ml, "cond", args)
    add_output(n, latent, args)
    return n


def build_i2v(args, client) -> dict:
    if not args.image:
        sys.exit("[错误] i2v 需要 --image")
    img = client.upload_file(args.image)
    n = base_nodes(args)
    m = TASK_MODELS["i2v"]
    mh, ml = add_dual_model(n, args, args.unet_high or m["high"], args.unet_low or m["low"])
    w, h = _wh(args)
    n["load_img"] = {"class_type": "LoadImage", "inputs": {"image": img}}
    n["cond"] = {"class_type": "WanImageToVideo", "inputs": {
        "positive": ["pos", 0], "negative": ["neg", 0], "vae": ["vae", 0],
        "width": w, "height": h, "length": args.length, "batch_size": 1,
        "start_image": ["load_img", 0]}}
    latent = add_dual_sampler(n, mh, ml, "cond", args)
    add_output(n, latent, args)
    return n


def build_flf2v(args, client) -> dict:
    if not (args.start and args.end):
        sys.exit("[错误] flf2v 需要 --start 和 --end")
    s = client.upload_file(args.start); e = client.upload_file(args.end)
    n = base_nodes(args)
    m = TASK_MODELS["flf2v"]
    mh, ml = add_dual_model(n, args, args.unet_high or m["high"], args.unet_low or m["low"])
    w, h = _wh(args)
    n["load_s"] = {"class_type": "LoadImage", "inputs": {"image": s}}
    n["load_e"] = {"class_type": "LoadImage", "inputs": {"image": e}}
    n["cond"] = {"class_type": "WanFirstLastFrameToVideo", "inputs": {
        "positive": ["pos", 0], "negative": ["neg", 0], "vae": ["vae", 0],
        "width": w, "height": h, "length": args.length, "batch_size": 1,
        "start_image": ["load_s", 0], "end_image": ["load_e", 0]}}
    latent = add_dual_sampler(n, mh, ml, "cond", args)
    add_output(n, latent, args)
    return n


def build_fun_inpaint(args, client) -> dict:
    if not (args.start and args.end):
        sys.exit("[错误] fun_inpaint 需要 --start 和 --end")
    s = client.upload_file(args.start); e = client.upload_file(args.end)
    n = base_nodes(args)
    m = TASK_MODELS["fun_inpaint"]
    mh, ml = add_dual_model(n, args, args.unet_high or m["high"], args.unet_low or m["low"])
    w, h = _wh(args)
    n["load_s"] = {"class_type": "LoadImage", "inputs": {"image": s}}
    n["load_e"] = {"class_type": "LoadImage", "inputs": {"image": e}}
    n["cond"] = {"class_type": "WanFunInpaintToVideo", "inputs": {
        "positive": ["pos", 0], "negative": ["neg", 0], "vae": ["vae", 0],
        "width": w, "height": h, "length": args.length, "batch_size": 1,
        "start_image": ["load_s", 0], "end_image": ["load_e", 0]}}
    latent = add_dual_sampler(n, mh, ml, "cond", args)
    add_output(n, latent, args)
    return n


def build_fun_control(args, client) -> dict:
    if not args.control_video:
        sys.exit("[错误] fun_control 需要 --control-video")
    cv = client.upload_file(args.control_video, "control video")
    n = base_nodes(args)
    m = TASK_MODELS["fun_control"]
    mh, ml = add_dual_model(n, args, args.unet_high or m["high"], args.unet_low or m["low"])
    w, h = _wh(args)
    # 控制视频经 GetVideoComponents -> Canny 提取边缘(官方模板做法)
    n["load_cv"] = {"class_type": "LoadVideo", "inputs": {"file": cv}}
    n["vc"] = {"class_type": "GetVideoComponents", "inputs": {"video": ["load_cv", 0]}}
    n["canny"] = {"class_type": "Canny", "inputs": {
        "image": ["vc", 0], "low_threshold": 0.1, "high_threshold": 0.6}}
    cond_inputs = {
        "positive": ["pos", 0], "negative": ["neg", 0], "vae": ["vae", 0],
        "width": w, "height": h, "length": args.length, "batch_size": 1,
        "control_video": ["canny", 0]}
    if args.image:
        img = client.upload_file(args.image)
        n["load_img"] = {"class_type": "LoadImage", "inputs": {"image": img}}
        cond_inputs["ref_image"] = ["load_img", 0]
    n["cond"] = {"class_type": "Wan22FunControlToVideo", "inputs": cond_inputs}
    latent = add_dual_sampler(n, mh, ml, "cond", args)
    add_output(n, latent, args)
    return n


def build_vace(args, client) -> dict:
    """多参考 / 控制视频生视频。VACE 支持 reference_image + control_video + control_masks。
    多张参考用 --ref 多次传入,经 BatchImagesNode 合并。"""
    if not (args.ref or args.control_video):
        sys.exit("[错误] vace 需要至少一个 --ref 或 --control-video")
    n = base_nodes(args)
    m = TASK_MODELS["vace"]
    mh, ml = add_dual_model(n, args, args.unet_high or m["high"], args.unet_low or m["low"])
    w, h = _wh(args)
    cond_inputs = {
        "positive": ["pos", 0], "negative": ["neg", 0], "vae": ["vae", 0],
        "width": w, "height": h, "length": args.length, "batch_size": 1,
        "strength": args.strength}
    # 参考图(可多张):逐个 LoadImage,>1 张用 BatchImagesNode 合并
    if args.ref:
        ref_ids = []
        for i, rp in enumerate(args.ref):
            name = client.upload_file(rp, f"参考图{i}")
            nid = f"ref_{i}"
            n[nid] = {"class_type": "LoadImage", "inputs": {"image": name}}
            ref_ids.append(nid)
        cond_inputs["reference_image"] = _batch_images(n, ref_ids, "refbatch")
    # 控制视频(可选)
    if args.control_video:
        cv = client.upload_file(args.control_video, "control video")
        n["load_cv"] = {"class_type": "LoadVideo", "inputs": {"file": cv}}
        n["vc"] = {"class_type": "GetVideoComponents", "inputs": {"video": ["load_cv", 0]}}
        cond_inputs["control_video"] = ["vc", 0]
    n["cond"] = {"class_type": "WanVaceToVideo", "inputs": cond_inputs}
    latent = add_dual_sampler(n, mh, ml, "cond", args)
    # VACE 需要 TrimVideoLatent 去掉参考帧
    n["trim"] = {"class_type": "TrimVideoLatent", "inputs": {
        "samples": latent, "trim_amount": ["cond", 3]}}
    add_output(n, ["trim", 0], args)
    return n


def build_phantom(args, client) -> dict:
    """多主体参考:多张主体图喂给 WanPhantomSubjectToVideo 的 images(batch)。"""
    if not args.ref:
        sys.exit("[错误] phantom 需要至少一个 --ref(主体参考图)")
    n = base_nodes(args)
    m = TASK_MODELS["phantom"]
    mh, ml = add_dual_model(n, args, args.unet_high or m["high"], args.unet_low or m["low"])
    w, h = _wh(args)
    ref_ids = []
    for i, rp in enumerate(args.ref):
        name = client.upload_file(rp, f"主体{i}")
        nid = f"subj_{i}"
        n[nid] = {"class_type": "LoadImage", "inputs": {"image": name}}
        ref_ids.append(nid)
    images_ref = _batch_images(n, ref_ids, "subjbatch")
    n["cond"] = {"class_type": "WanPhantomSubjectToVideo", "inputs": {
        "positive": ["pos", 0], "negative": ["neg", 0], "vae": ["vae", 0],
        "width": w, "height": h, "length": args.length, "batch_size": 1,
        "images": images_ref}}
    # Phantom 输出 (positive, negative_text, negative_img_text, latent)
    # KSampler 的 negative 用 negative_img_text(slot 2),latent 是 slot 3
    steps = args.steps; boundary = max(1, steps // 2)
    n["ks_high"] = {"class_type": "KSamplerAdvanced", "inputs": {
        "model": mh, "add_noise": "enable", "noise_seed": args.seed, "steps": steps,
        "cfg": args.cfg, "sampler_name": args.sampler, "scheduler": args.scheduler,
        "start_at_step": 0, "end_at_step": boundary, "return_with_leftover_noise": "enable",
        "positive": ["cond", 0], "negative": ["cond", 2], "latent_image": ["cond", 3]}}
    n["ks_low"] = {"class_type": "KSamplerAdvanced", "inputs": {
        "model": ml, "add_noise": "disable", "noise_seed": args.seed, "steps": steps,
        "cfg": args.cfg, "sampler_name": args.sampler, "scheduler": args.scheduler,
        "start_at_step": boundary, "end_at_step": 10000, "return_with_leftover_noise": "disable",
        "positive": ["cond", 0], "negative": ["cond", 2], "latent_image": ["ks_high", 0]}}
    add_output(n, ["ks_low", 0], args)
    return n


def build_s2v(args, client) -> dict:
    """音频驱动 / 人物口型。需 ref_image + audio。"""
    if not (args.image and args.audio):
        sys.exit("[错误] s2v 需要 --image(参考人像)和 --audio")
    img = client.upload_file(args.image); aud = client.upload_file(args.audio, "音频")
    n = base_nodes(args)
    m = TASK_MODELS["s2v"]
    mh, ml = add_dual_model(n, args, args.unet_high or m["high"], args.unet_low or m["low"])
    w, h = _wh(args)
    n["load_img"] = {"class_type": "LoadImage", "inputs": {"image": img}}
    n["load_aud"] = {"class_type": "LoadAudio", "inputs": {"audio": aud}}
    n["aenc_load"] = {"class_type": "AudioEncoderLoader", "inputs": {
        "audio_encoder_name": args.audio_encoder}}
    n["aenc"] = {"class_type": "AudioEncoderEncode", "inputs": {
        "audio_encoder": ["aenc_load", 0], "audio": ["load_aud", 0]}}
    n["cond"] = {"class_type": "WanSoundImageToVideo", "inputs": {
        "positive": ["pos", 0], "negative": ["neg", 0], "vae": ["vae", 0],
        "width": w, "height": h, "length": args.length, "batch_size": 1,
        "audio_encoder_output": ["aenc", 0], "ref_image": ["load_img", 0]}}
    latent = add_dual_sampler(n, mh, ml, "cond", args)
    add_output(n, latent, args, audio_ref=["load_aud", 0])
    return n


def build_animate(args, client) -> dict:
    """动作驱动 / 角色动画。需 reference_image + pose_video(姿态)。"""
    if not (args.image and args.pose_video):
        sys.exit("[错误] animate 需要 --image(角色参考)和 --pose-video(姿态视频)")
    img = client.upload_file(args.image); pv = client.upload_file(args.pose_video, "姿态视频")
    n = base_nodes(args)
    m = TASK_MODELS["animate"]
    mh, ml = add_dual_model(n, args, args.unet_high or m["high"], args.unet_low or m["low"])
    w, h = _wh(args)
    n["load_img"] = {"class_type": "LoadImage", "inputs": {"image": img}}
    n["load_pose"] = {"class_type": "LoadVideo", "inputs": {"file": pv}}
    n["pose_vc"] = {"class_type": "GetVideoComponents", "inputs": {"video": ["load_pose", 0]}}
    cond_inputs = {
        "positive": ["pos", 0], "negative": ["neg", 0], "vae": ["vae", 0],
        "width": w, "height": h, "length": args.length, "batch_size": 1,
        "continue_motion_max_frames": 77, "video_frame_offset": 0,
        "reference_image": ["load_img", 0], "pose_video": ["pose_vc", 0]}
    if args.face_video:
        fv = client.upload_file(args.face_video, "人脸视频")
        n["load_face"] = {"class_type": "LoadVideo", "inputs": {"file": fv}}
        n["face_vc"] = {"class_type": "GetVideoComponents", "inputs": {"video": ["load_face", 0]}}
        cond_inputs["face_video"] = ["face_vc", 0]
    n["cond"] = {"class_type": "WanAnimateToVideo", "inputs": cond_inputs}
    latent = add_dual_sampler(n, mh, ml, "cond", args)
    n["trim"] = {"class_type": "TrimVideoLatent", "inputs": {
        "samples": latent, "trim_amount": ["cond", 3]}}
    add_output(n, ["trim", 0], args)
    return n


BUILDERS = {
    "t2v": build_t2v, "i2v": build_i2v, "flf2v": build_flf2v,
    "fun_inpaint": build_fun_inpaint, "fun_control": build_fun_control,
    "vace": build_vace, "phantom": build_phantom, "s2v": build_s2v,
    "animate": build_animate,
}


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def print_models():
    print("各 task 默认所需权重(放进 models/diffusion_models/,lora 放 models/loras/):\n")
    print(f"  公共: text_encoders/{CLIP_NAME}  |  vae/{VAE_NAME}")
    print(f"  加速 lora(可选): loras/{LORA_HIGH}, loras/{LORA_LOW}\n")
    for t, m in TASK_MODELS.items():
        low = m.get("low") or "(单模型)"
        extra = ("  + " + m["extra"]) if m.get("extra") else ""
        print(f"  {t:12s} high={m['high']}\n               low={low}{extra}")


def parse_args():
    p = argparse.ArgumentParser(
        description="通过 ComfyUI API 用 Wan2.2 生成视频,覆盖 i2v/t2v/首尾帧/多参考/动作驱动/口型 等场景。",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("task", nargs="?", choices=list(BUILDERS), help="场景")
    p.add_argument("--list-models", action="store_true", help="列出每个 task 所需权重后退出")
    p.add_argument("--prompt", default="", help="正向提示词")
    p.add_argument("--negative", default=NEG_DEFAULT, help="负向提示词(默认用官方通用负面词)")
    p.add_argument("-o", "--output", default="wan_out.mp4", help="输出视频路径")
    # 输入素材
    p.add_argument("--image", help="输入图(i2v/s2v/animate/fun_control ref)")
    p.add_argument("--start", help="首帧(flf2v/fun_inpaint)")
    p.add_argument("--end", help="尾帧(flf2v/fun_inpaint)")
    p.add_argument("--ref", action="append", help="参考图(vace/phantom,可多次传入实现多参考)")
    p.add_argument("--control-video", dest="control_video", help="控制视频(vace/fun_control)")
    p.add_argument("--pose-video", dest="pose_video", help="姿态视频(animate)")
    p.add_argument("--face-video", dest="face_video", help="人脸视频(animate,可选)")
    p.add_argument("--audio", help="音频(s2v)")
    # 生成参数
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=640)
    p.add_argument("--length", type=int, default=81, help="帧数(默认 81)")
    p.add_argument("--fps", type=int, default=16, help="输出帧率(默认 16)")
    p.add_argument("--steps", type=int, default=20, help="采样步数(默认 20;配 lightx2v lora 可用 4)")
    p.add_argument("--cfg", type=float, default=3.5, help="CFG(默认 3.5;配加速 lora 用 1.0)")
    p.add_argument("--shift", type=float, default=8.0, help="ModelSamplingSD3 shift(默认 8.0)")
    p.add_argument("--sampler", default="euler", help="采样器(默认 euler;部分模板用 uni_pc)")
    p.add_argument("--scheduler", default="simple")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--strength", type=float, default=1.0, help="VACE 控制强度(默认 1.0)")
    # 模型覆盖
    p.add_argument("--unet-high", dest="unet_high", help="覆盖 high noise UNet 文件名")
    p.add_argument("--unet-low", dest="unet_low", help="覆盖 low noise UNet 文件名")
    p.add_argument("--clip", default=CLIP_NAME, help="text encoder 文件名")
    p.add_argument("--vae", default=VAE_NAME, help="VAE 文件名")
    p.add_argument("--lora-high", dest="lora_high", default="", help="high noise 加速 lora(默认不挂)")
    p.add_argument("--lora-low", dest="lora_low", default="", help="low noise 加速 lora(默认不挂)")
    p.add_argument("--audio-encoder", dest="audio_encoder",
                   default="wav2vec2_large_english_fp16.safetensors", help="s2v 音频编码器")
    p.add_argument("--server", default=DEFAULT_SERVER)

    args = p.parse_args()
    if args.list_models:
        print_models(); sys.exit(0)
    if not args.task:
        p.error("需要指定 task(或用 --list-models)")
    if not args.prompt and args.task != "s2v":
        print("[提示] 未提供 --prompt,将用空提示词生成。", file=sys.stderr)
    return args


def main():
    args = parse_args()
    client = ComfyClient(args.server)
    client.ping()

    print(f"[1/3] 构建 {args.task} 工作流...")
    prompt = BUILDERS[args.task](args, client)

    print(f"[2/3] 提交到 {args.server}(steps={args.steps}, {args.width}x{args.height}, {args.length}帧)...")
    pid = client.submit(prompt)
    print(f"      prompt_id={pid},等待生成(视频较慢,请耐心)...")

    outputs = client.wait(pid)
    saved = client.download(outputs, args.output)
    if saved:
        print(f"[3/3] 完成,已保存:")
        for s in saved:
            print(f"      {s}")
    else:
        print("[3/3] 执行成功但未抓到视频输出(检查工作流是否含 SaveVideo)。")


if __name__ == "__main__":
    main()

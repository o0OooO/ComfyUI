#!/usr/bin/env python3
"""
sensenova_compose.py — SenseNova-U1 多图参考融合("溶图")CLI。

传入多张参考图(角色/道具/场景...)+ 一句提示词,让模型把这些视觉概念
融合进同一张生成图。例如:"这两个角色拿着这个道具,在这个场景里干什么"。

与 sensenova_api.py 的区别:
  - sensenova_api.py 走 ComfyUI HTTP API,但 ComfyUI 的 SenseNova 编辑节点写死单图,
    无法多图融合。
  - 本脚本**直接调用底层 it2i_generate**(和官方 examples/editing/inference.py 同路径),
    原生支持多图 —— 每个 prompt 里的 <image> 占位符绑定一张输入图。

底层机制(已核实 modeling_neo_chat.py):
  image_token_count = prompt.count('<image>')
  assert len(images) >= image_token_count        # 图数必须 >= <image> 数
  if len(images) > image_token_count:             # 多余的自动补在开头
      prompt = "<image>\n" * (差额) + prompt
  逐张图编码后一起进模型融合。

⚠️ 必须在 conda `sensenova` 环境运行(transformers==4.57.1;5.x 跑不了该模型)。
   首次会加载 ~16GB 权重(已缓存则快)。

⚠️ 显存:多图 + 默认 2048² 输入/输出在 48GB L40S 上会 OOM(attention 随分辨率平方膨胀)。
   实测稳妥配置:--input-max-pixels 589824(768²)或 1048576(1024²) + 较小 --width/--height。
   图越多越要调小。设 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 可缓解碎片。

用法:
  # 显式用 <image> 占位绑定每张图(推荐,可控)
  python sensenova_compose.py \
    --image 角色A.png --image 角色B.png --image 道具.png --image 场景.png \
    --prompt "<image> 和 <image> 两人一起举着 <image>,站在 <image> 的场景中欢笑" \
    -o out.png

  # 不写占位符:模型自动在开头补齐 N 个 <image>,再接你的描述
  python sensenova_compose.py \
    --image a.png --image b.png --image prop.png \
    --prompt "these characters hold the item together in a park" -o out.png
"""
from __future__ import annotations

import argparse
import math
import os
import sys

# 确保能 import sensenova_u1(已 pip 安装则不需要;保险起见加 src 路径)
_SRC = os.environ.get("SENSENOVA_U1_SRC", "/home/ubuntu/SenseNova-U1/src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

DEFAULT_MODEL = "sensenova/SenseNova-U1-8B-MoT"
DEFAULT_SEED = 42
GRID = 32                     # 输出 H/W 必须是它的倍数(image-token grid)
TARGET_PIXELS = 2048 * 2048   # 自动分辨率的总像素目标


def _lazy_imports():
    import torch
    from PIL import Image
    import sensenova_u1
    from sensenova_u1.models.neo_unify.utils import smart_resize
    from sensenova_u1.utils import load_model_and_tokenizer, make_offload_ctx
    import numpy as np
    return torch, Image, sensenova_u1, smart_resize, load_model_and_tokenizer, make_offload_ctx, np


def load_rgb(path, smart_resize, input_max_pixels):
    """加载为 RGB(RGBA 贴白底),按预算缩放。"""
    from PIL import Image
    if not os.path.isfile(path):
        sys.exit(f"[错误] 找不到图片:{path}")
    img = Image.open(path)
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3]); img = bg
    img = img.convert("RGB")
    if input_max_pixels:
        h, w = smart_resize(height=img.height, width=img.width, factor=GRID,
                            min_pixels=input_max_pixels, max_pixels=input_max_pixels)
        if (w, h) != img.size:
            img = img.resize((w, h), Image.LANCZOS)
    return img


def resolve_output_size(first_img, smart_resize, explicit, target_pixels):
    if explicit:
        w, h = explicit
        if w % GRID or h % GRID:
            sys.exit(f"[错误] --width/--height 必须是 {GRID} 的倍数")
        return w, h
    h, w = smart_resize(height=first_img.height, width=first_img.width, factor=GRID,
                        min_pixels=target_pixels, max_pixels=target_pixels)
    return w, h


def auto_input_max_pixels(n):
    """官方策略:<=2 张保持 2048²,更多则在 2*2048² 总预算里均分。"""
    full = 2 * (2048 * 2048)
    if n <= 2:
        return 2048 * 2048
    return max(512 * 512, full // n)


def main():
    p = argparse.ArgumentParser(
        description="SenseNova-U1 多图参考融合(溶图):多张参考图 + 提示词 -> 一张融合图。",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", action="append", required=True,
                   help="参考图路径,多次传入传多张(角色/道具/场景...)")
    p.add_argument("--prompt", required=True,
                   help="提示词。用 <image> 占位符按顺序绑定每张图;不写则自动在开头补齐。")
    p.add_argument("-o", "--output", default="compose_out.png")
    p.add_argument("--width", type=int, help=f"输出宽(>{GRID}倍数);需与 --height 同给")
    p.add_argument("--height", type=int, help=f"输出高(>{GRID}倍数)")
    p.add_argument("--target-pixels", type=int, default=TARGET_PIXELS, dest="target_pixels")
    p.add_argument("--input-max-pixels", type=int, default=0, dest="input_max_pixels",
                   help="每张输入图的像素上限(默认 0=按图数自动)。多图易 OOM 时调小,如 1048576(1024²)")
    p.add_argument("--cfg-scale", type=float, default=4.0, dest="cfg_scale", help="文本 CFG(默认 4.0)")
    p.add_argument("--img-cfg-scale", type=float, default=1.0, dest="img_cfg_scale", help="图像 CFG(默认 1.0)")
    p.add_argument("--cfg-norm", default="none", dest="cfg_norm", choices=["none", "global", "channel"])
    p.add_argument("--timestep-shift", type=float, default=3.0, dest="timestep_shift")
    p.add_argument("--num-steps", type=int, default=50, dest="num_steps", help="采样步数(默认 50;预览用 2)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--think", action="store_true", help="开启思维链(先推理再生成,更慢)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = p.parse_args()

    if (args.width is None) != (args.height is None):
        sys.exit("[错误] --width 和 --height 必须同时给或同时不给")

    torch, Image, sensenova_u1, smart_resize, load_model_and_tokenizer, make_offload_ctx, np = _lazy_imports()
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    n = len(args.image)
    placeholders = args.prompt.count("<image>")
    if placeholders > n:
        sys.exit(f"[错误] prompt 里有 {placeholders} 个 <image>,但只传了 {n} 张图(图数必须 >= 占位符数)")

    imp = args.input_max_pixels if args.input_max_pixels else auto_input_max_pixels(n)
    print(f"[1/3] 加载 {n} 张参考图(每张上限约 {int(math.sqrt(imp))}px),融合占位符 {placeholders} 个...")
    images = [load_rgb(pth, smart_resize, imp) for pth in args.image]
    w, h = resolve_output_size(images[0], smart_resize,
                               (args.width, args.height) if args.width else None, args.target_pixels)

    print(f"[2/3] 加载模型 {args.model} 并生成({w}x{h}, steps={args.num_steps})...")
    model, tokenizer = load_model_and_tokenizer(args.model, dtype=dtype, device=args.device)

    NORM_MEAN = (0.5, 0.5, 0.5); NORM_STD = (0.5, 0.5, 0.5)
    def to_pil(batch):
        mean = torch.tensor(NORM_MEAN, device=batch.device, dtype=batch.dtype).view(1, 3, 1, 1)
        std = torch.tensor(NORM_STD, device=batch.device, dtype=batch.dtype).view(1, 3, 1, 1)
        arr = ((batch.float() * std + mean).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8)
        return [Image.fromarray(a) for a in arr]

    with torch.inference_mode():
        out = model.it2i_generate(
            tokenizer, args.prompt, list(images),
            image_size=(w, h), cfg_scale=args.cfg_scale, img_cfg_scale=args.img_cfg_scale,
            cfg_norm=args.cfg_norm, timestep_shift=args.timestep_shift,
            cfg_interval=(0.0, 1.0), num_steps=args.num_steps, batch_size=1,
            think_mode=args.think, seed=args.seed)
    think_text = ""
    if args.think:
        out, think_text = out[0], out[1]
    result = to_pil(out)

    result[0].save(args.output)
    print(f"[3/3] 完成,已保存 {args.output}")
    if think_text:
        tp = os.path.splitext(args.output)[0] + ".think.txt"
        open(tp, "w").write(think_text)
        print(f"      思维链 -> {tp}")


if __name__ == "__main__":
    main()

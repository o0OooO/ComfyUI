#!/usr/bin/env python3
"""
sensenova_api.py — 通过 ComfyUI HTTP API 驱动 SenseNova U1 完成各类图像任务的统一 CLI。

覆盖 Google Imagen 2(img-2)的全部图像能力,一个脚本按 --task 参数切换场景:

  t2i        文生图                （Imagen 2: text-to-image）
  edit       整图指令编辑（无蒙版）  （Imagen 2: mask-free edit）
  inpaint    蒙版局部重绘           （Imagen 2: inpainting，只改蒙版区，其余保留）
  outpaint   向外扩图               （Imagen 2: outpainting）
  interleave 图文交错生成 + 思维链    （SenseNova 额外能力）
  compose    多图参考融合(溶图)      （多张角色/道具/场景图 + 提示词 -> 一张融合图）

它在脚本内**程序化构建 ComfyUI API prompt**(不依赖磁盘上的 workflow JSON),
所以不用为每个场景手动加载不同的 .json —— 场景与参数都由命令行决定。

依赖:仅标准库(urllib/json/argparse)。需要一个正在运行的 ComfyUI 服务
(本机或容器),且已安装 SenseNova U1 节点。

示例:
  # 文生图
  python sensenova_api.py t2i --prompt "a red panda on a skateboard" --ratio 16:9 -o out.png

  # 整图编辑(无蒙版)
  python sensenova_api.py edit --image cat.png --prompt "make it wear sunglasses" -o out.png

  # 蒙版局部重绘:在 input 目录放一张带 alpha 通道的 PNG(透明=要改的区域),
  # 或用 --mask 指定独立的灰度蒙版图(白=改,黑=留)
  python sensenova_api.py inpaint --image room.png --mask wall_mask.png \
      --prompt "a window with ocean view" -o out.png

  # 扩图(四周各扩 256px)
  python sensenova_api.py outpaint --image photo.png --pad 256 \
      --prompt "extend the beach scene naturally" -o out.png

  # 图文交错
  python sensenova_api.py interleave --prompt "explain photosynthesis with diagrams" -o out.png

  # 多图融合(溶图):多张参考图 + <image> 占位符按序绑定(最多 6 张)
  python sensenova_api.py compose \
      --ref charA.png --ref charB.png --ref prop.png --ref scene.png \
      --prompt "<image> 和 <image> 拿着 <image>,站在 <image> 的场景里庆祝" \
      --input-mp 1.0 -o out.png
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

# ----------------------------------------------------------------------------
# 常量(取自 SenseNova U1 节点源码,已核实)
# ----------------------------------------------------------------------------
DEFAULT_SERVER = os.environ.get("COMFYUI_SERVER", "http://127.0.0.1:8188")
DEFAULT_MODEL = "sensenova/SenseNova-U1-8B-MoT"
DEFAULT_SEED = 42

# t2i 节点用 "WxH|ratio" 字符串
T2I_RESOLUTIONS = {
    "1:1": "2048x2048|1:1", "16:9": "2720x1536|16:9", "9:16": "1536x2720|9:16",
    "3:2": "2496x1664|3:2", "2:3": "1664x2496|2:3", "4:3": "2368x1760|4:3",
    "3:4": "1760x2368|3:4", "1:2": "1440x2880|1:2", "2:1": "2880x1440|2:1",
    "1:3": "1152x3456|1:3", "3:1": "3456x1152|3:1",
}
INTERLEAVE_RESOLUTIONS = {
    "1:1": "1536x1536|1:1", "16:9": "2048x1152|16:9", "9:16": "1152x2048|9:16",
    "3:2": "1888x1248|3:2",
}


# ----------------------------------------------------------------------------
# ComfyUI API 客户端
# ----------------------------------------------------------------------------
class ComfyClient:
    def __init__(self, server: str):
        self.server = server.rstrip("/")
        self.client_id = uuid.uuid4().hex

    def _post(self, path: str, payload: dict) -> dict:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.server}{path}", data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())

    def _get(self, path: str) -> dict:
        with urllib.request.urlopen(f"{self.server}{path}", timeout=60) as r:
            return json.loads(r.read())

    def ping(self) -> None:
        try:
            self._get("/system_stats")
        except Exception as e:
            sys.exit(f"[错误] 无法连接 ComfyUI 服务 {self.server}:{e}\n"
                     f"      请确认服务在运行(本机 python main.py,或容器已启动并映射端口)。")

    def upload_image(self, filepath: str, subfolder: str = "") -> str:
        """上传本地图片到 ComfyUI 的 input 目录,返回服务端文件名(供 LoadImage 引用)。"""
        if not os.path.isfile(filepath):
            sys.exit(f"[错误] 找不到输入图片:{filepath}")
        filename = os.path.basename(filepath)
        boundary = "----sensenova" + uuid.uuid4().hex
        with open(filepath, "rb") as f:
            content = f.read()
        body = []
        body.append(f"--{boundary}\r\n".encode())
        body.append(
            f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
            f"Content-Type: application/octet-stream\r\n\r\n".encode())
        body.append(content)
        body.append(f"\r\n--{boundary}\r\n".encode())
        body.append('Content-Disposition: form-data; name="overwrite"\r\n\r\n'.encode())
        body.append(b"true")
        body.append(f"\r\n--{boundary}--\r\n".encode())
        data = b"".join(body)
        req = urllib.request.Request(
            f"{self.server}/upload/image", data=data,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            resp = json.loads(r.read())
        name = resp.get("name", filename)
        sub = resp.get("subfolder", "")
        return f"{sub}/{name}" if sub else name

    def submit(self, prompt: dict) -> str:
        resp = self._post("/prompt", {"prompt": prompt, "client_id": self.client_id})
        if resp.get("node_errors"):
            sys.exit(f"[错误] 节点校验失败:\n{json.dumps(resp['node_errors'], ensure_ascii=False, indent=2)}")
        return resp["prompt_id"]

    def wait(self, prompt_id: str, poll: float = 2.0, timeout: float = 1800) -> dict:
        """轮询 history 直到该 prompt 完成,返回 outputs。"""
        start = time.time()
        while True:
            hist = self._get(f"/history/{prompt_id}")
            if prompt_id in hist:
                info = hist[prompt_id]
                status = info.get("status", {})
                if status.get("completed") or status.get("status_str") == "success":
                    return info.get("outputs", {})
                if status.get("status_str") == "error":
                    msgs = status.get("messages", [])
                    sys.exit(f"[错误] 执行失败:\n{json.dumps(msgs, ensure_ascii=False, indent=2)}")
            if time.time() - start > timeout:
                sys.exit(f"[错误] 等待超时({timeout}s),prompt_id={prompt_id}")
            time.sleep(poll)

    def download_images(self, outputs: dict, out_path: str) -> list[str]:
        """从 outputs 里抓所有 SaveImage/PreviewImage 产生的图片,保存到本地。"""
        saved = []
        idx = 0
        base, ext = os.path.splitext(out_path)
        ext = ext or ".png"
        for node_id, out in outputs.items():
            for im in out.get("images", []):
                params = urllib.parse.urlencode({
                    "filename": im["filename"],
                    "subfolder": im.get("subfolder", ""),
                    "type": im.get("type", "output"),
                })
                with urllib.request.urlopen(f"{self.server}/view?{params}", timeout=120) as r:
                    data = r.read()
                # 跳过 1x1 占位图(interleave 在纯文本输出时会产生)
                if len(data) < 256:
                    continue
                target = out_path if idx == 0 else f"{base}_{idx}{ext}"
                with open(target, "wb") as f:
                    f.write(data)
                saved.append(target)
                idx += 1
        # 抓取文本输出(interleave 的 markdown / 编辑的 think_text)存成 .txt
        texts = []
        for node_id, out in outputs.items():
            for key in ("text", "markdown", "string"):
                for t in out.get(key, []) if isinstance(out.get(key), list) else []:
                    if isinstance(t, str) and t.strip():
                        texts.append(t)
        if texts:
            txt_path = f"{base}.txt"
            with open(txt_path, "w") as f:
                f.write("\n\n".join(texts))
            saved.append(txt_path)
        return saved


# ----------------------------------------------------------------------------
# Prompt 构建:每个场景一个函数,返回 ComfyUI API prompt dict
# ----------------------------------------------------------------------------
def loader_node(args) -> dict:
    return {"class_type": "SenseNovaU1LocalLoader", "inputs": {
        "model_path": args.model, "sensenova_u1_src": "", "device": args.device,
        "dtype": args.dtype, "attn_backend": "auto", "device_map": "none",
        "max_memory": "", "vram_mode": args.vram_mode, "gguf_checkpoint": ""}}


def build_t2i(args, client) -> dict:
    res = T2I_RESOLUTIONS.get(args.ratio)
    if res is None:
        sys.exit(f"[错误] t2i 不支持的比例 {args.ratio},可选:{', '.join(T2I_RESOLUTIONS)}")
    return {
        "1": loader_node(args),
        "2": {"class_type": "SenseNovaU1LocalTextToImage", "inputs": {
            "u1_model": ["1", 0], "prompt": args.prompt, "resolution": res,
            "cfg_scale": args.cfg_scale, "cfg_norm": "none", "timestep_shift": 3.0,
            "cfg_interval_start": 0.0, "cfg_interval_end": 1.0,
            "num_steps": args.steps, "batch_size": args.batch, "seed": args.seed,
            "think_mode": args.think}},
        "3": {"class_type": "SaveImage", "inputs": {
            "images": ["2", 0], "filename_prefix": "SenseNova_t2i"}},
    }


def edit_common(args, client):
    """上传图片,返回服务端文件名。"""
    return client.upload_image(args.image)


def build_edit(args, client) -> dict:
    img_name = edit_common(args, client)
    return {
        "1": loader_node(args),
        "2": {"class_type": "LoadImage", "inputs": {"image": img_name}},
        "3": {"class_type": "SenseNovaU1LocalImageEdit", "inputs": {
            "u1_model": ["1", 0], "image": ["2", 0], "prompt": args.prompt,
            "auto_size": True, "width": 2048, "height": 2048, "target_megapixels": 4.194304,
            "cfg_scale": args.cfg_scale, "img_cfg_scale": args.img_cfg_scale,
            "cfg_norm": "none", "timestep_shift": 3.0,
            "cfg_interval_start": 0.0, "cfg_interval_end": 1.0,
            "num_steps": args.steps, "batch_size": args.batch, "seed": args.seed,
            "think_mode": args.think}},
        "4": {"class_type": "SaveImage", "inputs": {
            "images": ["3", 0], "filename_prefix": "SenseNova_edit"}},
    }


def build_inpaint(args, client) -> dict:
    """蒙版局部重绘:SenseNova 整图重绘 -> 仅蒙版区合成回原图。
    蒙版来源:--mask 独立灰度图(白=改);否则用 LoadImage 输出的 alpha 通道。"""
    img_name = client.upload_image(args.image)
    nodes = {
        "1": loader_node(args),
        "2": {"class_type": "LoadImage", "inputs": {"image": img_name}},
        "3": {"class_type": "SenseNovaU1LocalImageEdit", "inputs": {
            "u1_model": ["1", 0], "image": ["2", 0], "prompt": args.prompt,
            "auto_size": True, "width": 2048, "height": 2048, "target_megapixels": 4.194304,
            "cfg_scale": args.cfg_scale, "img_cfg_scale": args.img_cfg_scale,
            "cfg_norm": "none", "timestep_shift": 3.0,
            "cfg_interval_start": 0.0, "cfg_interval_end": 1.0,
            "num_steps": args.steps, "batch_size": 1, "seed": args.seed,
            "think_mode": args.think}},
    }
    # 蒙版来源
    if args.mask:
        mask_name = client.upload_image(args.mask)
        nodes["10"] = {"class_type": "LoadImage", "inputs": {"image": mask_name}}
        # 用红通道转 mask(白=1)
        nodes["11"] = {"class_type": "ImageToMask", "inputs": {"image": ["10", 0], "channel": "red"}}
        mask_src = ["11", 0]
    else:
        # 用原图的 alpha 通道(LoadImage 第二输出);透明区=要编辑
        mask_src = ["2", 1]
    # 扩张 + 羽化,柔化边缘
    nodes["4"] = {"class_type": "GrowMask", "inputs": {
        "mask": mask_src, "expand": args.grow, "tapered_corners": True}}
    nodes["5"] = {"class_type": "FeatherMask", "inputs": {
        "mask": ["4", 0], "left": args.feather, "top": args.feather,
        "right": args.feather, "bottom": args.feather}}
    # 合成:dest=原图, source=编辑结果(自动缩放对齐), mask=羽化蒙版
    nodes["6"] = {"class_type": "ImageCompositeMasked", "inputs": {
        "destination": ["2", 0], "source": ["3", 0], "x": 0, "y": 0,
        "resize_source": True, "mask": ["5", 0]}}
    nodes["7"] = {"class_type": "SaveImage", "inputs": {
        "images": ["6", 0], "filename_prefix": "SenseNova_inpaint"}}
    return nodes


def build_outpaint(args, client) -> dict:
    """扩图:pad 画布 -> SenseNova 续画 -> 仅扩展区合成(中心保留原图)。"""
    img_name = client.upload_image(args.image)
    p = args.pad
    return {
        "1": loader_node(args),
        "2": {"class_type": "LoadImage", "inputs": {"image": img_name}},
        "3": {"class_type": "ImagePadForOutpaint", "inputs": {
            "image": ["2", 0], "left": p, "top": p, "right": p, "bottom": p,
            "feathering": args.feather}},
        "4": {"class_type": "SenseNovaU1LocalImageEdit", "inputs": {
            "u1_model": ["1", 0], "image": ["3", 0], "prompt": args.prompt,
            "auto_size": True, "width": 2048, "height": 2048, "target_megapixels": 4.194304,
            "cfg_scale": args.cfg_scale, "img_cfg_scale": args.img_cfg_scale,
            "cfg_norm": "none", "timestep_shift": 3.0,
            "cfg_interval_start": 0.0, "cfg_interval_end": 1.0,
            "num_steps": args.steps, "batch_size": 1, "seed": args.seed,
            "think_mode": args.think}},
        # pad mask: 扩展区=1,直接当合成 mask
        "5": {"class_type": "ImageCompositeMasked", "inputs": {
            "destination": ["3", 0], "source": ["4", 0], "x": 0, "y": 0,
            "resize_source": True, "mask": ["3", 1]}},
        "6": {"class_type": "SaveImage", "inputs": {
            "images": ["5", 0], "filename_prefix": "SenseNova_outpaint"}},
    }


def build_interleave(args, client) -> dict:
    res = INTERLEAVE_RESOLUTIONS.get(args.ratio, INTERLEAVE_RESOLUTIONS["1:1"])
    nodes = {
        "1": loader_node(args),
        "2": {"class_type": "SenseNovaU1LocalInterleave", "inputs": {
            "u1_model": ["1", 0], "prompt": args.prompt, "resolution": res,
            "system_message": "", "cfg_scale": args.cfg_scale, "img_cfg_scale": args.img_cfg_scale,
            "timestep_shift": 3.0, "cfg_interval_start": 0.0, "cfg_interval_end": 1.0,
            "num_steps": args.steps, "seed": args.seed, "think_mode": args.think}},
        # interleave 的主要价值是图文交错:SaveImage 抓图,Preview 节点输出 markdown 文本
        "3": {"class_type": "SaveImage", "inputs": {
            "images": ["2", 0], "filename_prefix": "SenseNova_interleave"}},
        "4": {"class_type": "SenseNovaInterleavePreview", "inputs": {
            "interleave_result": ["2", 4], "include_think": args.think, "images": ["2", 0]}},
    }
    if args.image:
        img_name = client.upload_image(args.image)
        nodes["5"] = {"class_type": "LoadImage", "inputs": {"image": img_name}}
        nodes["2"]["inputs"]["image"] = ["5", 0]
    return nodes


def build_compose(args, client) -> dict:
    """多图参考融合(溶图):多张参考图(角色/道具/场景)+ prompt(<image> 占位符按序绑定)
    -> 一张融合图。走 SenseNovaU1LocalCompose 节点(image + image2~6,最多 6 张)。"""
    refs = args.ref or ([args.image] if args.image else [])
    if not refs:
        sys.exit("[错误] compose 需要至少一个 --ref(或 --image)参考图")
    if len(refs) > 6:
        sys.exit(f"[错误] compose 最多 6 张参考图,收到 {len(refs)} 张")
    placeholders = args.prompt.count("<image>")
    if placeholders > len(refs):
        sys.exit(f"[错误] prompt 里有 {placeholders} 个 <image>,但只传了 {len(refs)} 张参考图")

    nodes = {"1": loader_node(args)}
    # 每张参考图一个 LoadImage,映射到 compose 节点的 image / image2..image6
    slot_names = ["image", "image2", "image3", "image4", "image5", "image6"]
    compose_inputs = {
        "u1_model": ["1", 0], "prompt": args.prompt,
        "auto_size": args.width == 0 or args.height == 0,
        "width": args.width or 1024, "height": args.height or 1024,
        "target_megapixels": args.target_mp, "input_megapixels": args.input_mp,
        "cfg_scale": args.cfg_scale, "img_cfg_scale": args.img_cfg_scale,
        "cfg_norm": "none", "timestep_shift": 3.0,
        "cfg_interval_start": 0.0, "cfg_interval_end": 1.0,
        "num_steps": args.steps, "seed": args.seed, "think_mode": args.think}
    for i, rp in enumerate(refs):
        name = client.upload_image(rp)
        nid = f"img_{i}"
        nodes[nid] = {"class_type": "LoadImage", "inputs": {"image": name}}
        compose_inputs[slot_names[i]] = [nid, 0]
    nodes["compose"] = {"class_type": "SenseNovaU1LocalCompose", "inputs": compose_inputs}
    nodes["save"] = {"class_type": "SaveImage", "inputs": {
        "images": ["compose", 0], "filename_prefix": "SenseNova_compose"}}
    return nodes


BUILDERS = {
    "t2i": build_t2i, "edit": build_edit, "inpaint": build_inpaint,
    "outpaint": build_outpaint, "interleave": build_interleave,
    "compose": build_compose,
}


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="通过 ComfyUI API 用 SenseNova U1 覆盖 Imagen 2 的图像能力。",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("task", choices=list(BUILDERS), help="场景:t2i/edit/inpaint/outpaint/interleave/compose")
    p.add_argument("--prompt", required=True, help="文本指令/描述。compose 用 <image> 占位符按序绑定每张参考图")
    p.add_argument("-o", "--output", default="sensenova_out.png", help="输出图片路径(默认 sensenova_out.png)")
    p.add_argument("--image", help="输入图片(edit/inpaint/outpaint 必需;interleave 可选)")
    p.add_argument("--ref", action="append", help="参考图(compose 多图融合,可多次传入,最多 6 张)")
    p.add_argument("--mask", help="inpaint 用的独立灰度蒙版图(白=改,黑=留)。不给则用 --image 的 alpha 通道")
    p.add_argument("--ratio", default="1:1", help="t2i/interleave 输出比例(默认 1:1)")
    p.add_argument("--pad", type=int, default=256, help="outpaint 四周扩展像素(默认 256)")
    p.add_argument("--grow", type=int, default=6, help="inpaint 蒙版扩张像素(默认 6)")
    p.add_argument("--feather", type=int, default=24, help="蒙版/扩图羽化像素(默认 24)")
    p.add_argument("--steps", type=int, default=50, help="采样步数(默认 50;快速预览用 2)")
    p.add_argument("--batch", type=int, default=1, help="批量数(仅 t2i/edit,默认 1)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"随机种子(默认 {DEFAULT_SEED})")
    p.add_argument("--width", type=int, default=0, help="compose 输出宽(0=auto;32 的倍数)")
    p.add_argument("--height", type=int, default=0, help="compose 输出高(0=auto;32 的倍数)")
    p.add_argument("--target-mp", type=float, default=1.048576, dest="target_mp", help="compose 输出像素预算 MP(默认 1.0=1024²)")
    p.add_argument("--input-mp", type=float, default=1.048576, dest="input_mp", help="compose 每张输入图像素上限 MP(默认 1.0;多图易 OOM 时调小)")
    p.add_argument("--cfg-scale", type=float, default=4.0, dest="cfg_scale", help="文本 CFG(默认 4.0)")
    p.add_argument("--img-cfg-scale", type=float, default=1.0, dest="img_cfg_scale", help="图像 CFG(默认 1.0)")
    p.add_argument("--think", action="store_true", help="开启思维链(更慢,适合复杂推理编辑)")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"模型(默认 {DEFAULT_MODEL})")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--vram-mode", default="full", dest="vram_mode", choices=["full", "low", "balanced"])
    p.add_argument("--server", default=DEFAULT_SERVER, help=f"ComfyUI 服务地址(默认 {DEFAULT_SERVER})")
    args = p.parse_args()

    # 场景必需参数校验
    if args.task in ("edit", "inpaint", "outpaint") and not args.image:
        p.error(f"--task {args.task} 需要 --image")
    return args


def main():
    args = parse_args()
    client = ComfyClient(args.server)
    client.ping()

    print(f"[1/3] 构建 {args.task} 工作流...")
    prompt = BUILDERS[args.task](args, client)

    print(f"[2/3] 提交到 {args.server} ...")
    prompt_id = client.submit(prompt)
    print(f"      prompt_id={prompt_id},等待执行(steps={args.steps})...")

    outputs = client.wait(prompt_id)
    saved = client.download_images(outputs, args.output)

    if saved:
        print(f"[3/3] 完成,已保存 {len(saved)} 张:")
        for s in saved:
            print(f"      {s}")
    else:
        print("[3/3] 执行成功,但没有抓到图片输出(检查工作流是否含 SaveImage)。")


if __name__ == "__main__":
    main()

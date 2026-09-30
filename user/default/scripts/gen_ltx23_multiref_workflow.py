#!/usr/bin/env python3
"""Generate an LTX-2.3 multi-reference ("Ingredients" IC-LoRA) ComfyUI workflow.

Mirrors Lightricks' official LTX-2.3_ICLoRA_Ingredients_Single_Stage_Distilled.json
using only core ComfyUI nodes (no ComfyUI-LTXVideo pack required).
"""
import json
import os
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))          # .../ComfyUI
OUT = os.path.join(REPO, "user/default/workflows/video_ltx2_3_multiref_ingredients.json")

# 节点 schema 从运行中的 ComfyUI 拉(校验节点名/输入名/类型全靠它)。
# 需要先启动服务:bash user/default/scripts/start_comfy.sh
CACHE = "/tmp/comfy_object_info.json"
try:
    with urllib.request.urlopen("http://127.0.0.1:8188/object_info", timeout=30) as r:
        OI = json.load(r)
    with open(CACHE, "w") as f:
        json.dump(OI, f)
except Exception as e:
    if not os.path.exists(CACHE):
        raise SystemExit(
            f"拿不到 /object_info ({e})。先启动 ComfyUI:\n"
            f"  bash user/default/scripts/start_comfy.sh"
        )
    print(f"! /object_info 不可用 ({e}),用缓存 {CACHE}")
    OI = json.load(open(CACHE))

W, H, FRAMES, FPS = 768, 448, 121, 24
CELL_W, CELL_H = W // 2, H // 2
CKPT = "ltx-2.3-22b-dev-fp8.safetensors"
TE = "gemma_3_12B_it_fp4_mixed.safetensors"
DISTILL_LORA = "ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors"
ING_LORA = "ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors"
SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"

POSITIVE = """### Reference Sheet Description
**Top Row Left (Character):** <describe protagonist A - face close-up plus body turnaround: hair, eyes, skin tone, full outfit with colours and materials.>
**Top Row Right (Character):** <describe protagonist B the same way, or a second view of A.>
**Bottom Row Left (Prop):** <describe the key prop as a product-style render, multiple angles.>
**Bottom Row Right (Setting):** <describe the location: architecture, lighting, background depth.>

### Target Description
<Describe the shot you want. Name every element exactly as described above so the model
binds them to the sheet. Cover, in order: shot size and camera move, who is present and
what they wear, the action beat by beat, spoken dialogue in quotes, then lighting and
ambient audio.>"""

NEGATIVE = "worst quality, inconsistent motion, blurry, jittery, distorted"

NOTE_MAIN = f"""# LTX-2.3 多参考图生视频 (Ingredients IC-LoRA)

## 原理
LTX-2.3 的多参考**不是**喂多张独立图片,而是把所有主角/道具/场景拼成**一张 reference sheet**
(参考图册),再把这张图 loop 成一段静态视频当 in-context 条件。左边 4 个 `Load Image`
会自动拼成 2x2 的 {W}x{H} 图册。

## 用前必做
1. 下载 IC-LoRA (**需要先在 HF 页面点同意授权**):
   https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients
   放到 `models/loras/{ING_LORA}`
2. 4 张参考图分别放:主角A / 主角B / 关键道具 / 场景。
   - 参考图最好**黑底、无文字**,主体占满画面。
   - 只有 2-3 张参考时:把用不到的那路 `Image Stitch` 用 Ctrl+B 旁路掉。

## 提示词写法(很关键)
必须用两段式结构,和训练时一致:
- `### Reference Sheet Description` — 逐格描述图册里有什么(外观)
- `### Target Description` — 描述要生成的动作/镜头(叙事)

图册决定"长什么样",Target 决定"发生什么"。Target 里要**用图册里同样的措辞**重复
一遍角色和道具,模型才会把它们绑定到参考上。

## 效果调优
| 症状 | 处理 |
|---|---|
| 角色脸/衣服漂移 | 该角色的格子给正脸特写+转身图,别塞太满;IC-LoRA 强度提到 1.2-1.4 |
| 某个道具没出现 | 图册里必须有独立一格,并且在 `Reference sheet` 段落里写出来 |
| 参考跟得太死、不动 | `LTXVAddGuide` 的 strength 降到 0.8-0.9 |
| 想更高保真 | 参考格子给大一点(重要元素占的面积越大,还原越好) |

## 分辨率与长度
{W}x{H} / {FRAMES} 帧 / {FPS}fps 是官方训练 bucket,**效果最好,别乱改**。
宽高需能被 64 整除,帧数需为 8n+1。参考视频帧数必须 >= 121。

改分辨率时要同步改 4 处:`Empty LTXV Latent Video`、`Repeat Image Batch` 的 amount、
`LTXV Empty Latent Audio` 的 frames_number、以及 4 个 `Resize and Pad Image` 的尺寸
(各为总尺寸的一半)。

## 当前是蒸馏 8 步档 (快)
`ManualSigmas` 8 步 + cfg=1 + distilled LoRA 0.5。
想要**更高质量**:把 distilled LoRA 旁路(Ctrl+B),`CFGGuider` 的 cfg 改 4.0,
`ManualSigmas` 换成 30 步线性,例如:
`1.0, 0.966, 0.933, 0.9, 0.866, 0.833, 0.8, 0.766, 0.733, 0.7, 0.666, 0.633, 0.6, 0.566, 0.533, 0.5, 0.466, 0.433, 0.4, 0.366, 0.333, 0.3, 0.266, 0.233, 0.2, 0.166, 0.133, 0.1, 0.066, 0.033, 0.0`

## 显存
22B fp8 + 双 LoRA + 参考 token 翻倍,{W}x{H}x{FRAMES} 在 46GB L40S 上跑得动,
但要先确保 GPU 是空的。"""

NOTE_SHEET = """### 参考图册 (2x2)
拼好的图册会在这里预览。
先跑一次确认拼图正确、主体没被裁掉,再去调提示词。

布局:
```
+-----------+-----------+
| 主角 A    | 主角 B    |
+-----------+-----------+
| 关键道具  | 场景      |
+-----------+-----------+
```"""

nodes, links = [], []
_lid = [0]


def add(nid, ntype, pos, widgets=None, size=None, title=None):
    n = {
        "id": nid, "type": ntype, "pos": list(pos),
        "size": list(size) if size else [330, 120],
        "flags": {}, "order": len(nodes), "mode": 0,
        "inputs": [], "outputs": [], "properties": {"Node name for S&R": ntype},
    }
    if widgets is not None:
        n["widgets_values"] = widgets
    if title:
        n["title"] = title
    if ntype in OI:
        for name, typ in zip(OI[ntype]["output_name"], OI[ntype]["output"]):
            n["outputs"].append({"name": name, "type": typ, "links": [], "slot_index": len(n["outputs"])})
    nodes.append(n)
    return n


def byid(nid):
    return next(n for n in nodes if n["id"] == nid)


def link(src_id, src_slot, dst_id, dst_name):
    """Connect src output slot -> dst input by name, validating against object_info."""
    s, d = byid(src_id), byid(dst_id)
    typ = s["outputs"][src_slot]["type"]
    spec = OI[d["type"]]["input"]
    fields = {**spec.get("required", {}), **spec.get("optional", {})}
    assert dst_name in fields, f"{d['type']} has no input {dst_name}"
    want = fields[dst_name][0] if isinstance(fields[dst_name], list) else fields[dst_name]
    if isinstance(want, str) and want not in ("COMFY_MATCHTYPE_V3", "*"):
        assert want == typ, f"type mismatch {s['type']}[{src_slot}]:{typ} -> {d['type']}.{dst_name}:{want}"
    _lid[0] += 1
    lid = _lid[0]
    links.append([lid, src_id, src_slot, dst_id, fields and len(d["inputs"]) or 0, typ])
    links[-1][4] = len(d["inputs"])
    d["inputs"].append({"name": dst_name, "type": typ, "link": lid})
    s["outputs"][src_slot]["links"].append(lid)
    return lid


# ---- reference sheet builders ----
SLOTS = [("主角 A / Character A", 1), ("主角 B / Character B", 2),
         ("关键道具 / Key prop", 3), ("场景 / Location", 4)]
for i, (title, nid) in enumerate(SLOTS):
    add(nid, "LoadImage", (-1900, -600 + i * 400), ["example.png", "image"],
        size=[300, 340], title=f"参考 {nid}: {title}")
    add(10 + nid, "ResizeAndPadImage", (-1540, -600 + i * 400),
        [CELL_W, CELL_H, "black", "lanczos"], size=[300, 130],
        title=f"格子 {nid} -> {CELL_W}x{CELL_H}")
    link(nid, 0, 10 + nid, "image")

add(21, "ImageStitch", (-1180, -560), ["right", True, 0, "black"], size=[280, 130], title="上排: A | B")
link(11, 0, 21, "image1")
link(12, 0, 21, "image2")

add(22, "ImageStitch", (-1180, 240), ["right", True, 0, "black"], size=[280, 130], title="下排: 道具 | 场景")
link(13, 0, 22, "image1")
link(14, 0, 22, "image2")

add(23, "ImageStitch", (-850, -160), ["down", True, 0, "black"], size=[280, 130],
    title=f"拼成图册 {W}x{H}")
link(21, 0, 23, "image1")
link(22, 0, 23, "image2")

add(24, "PreviewImage", (-850, 60), [], size=[300, 330], title="图册预览")
link(23, 0, 24, "images")

add(30, "RepeatImageBatch", (-850, 440), [FRAMES], size=[300, 90],
    title=f"图册 -> {FRAMES} 帧静态视频")
link(23, 0, 30, "image")

# ---- loaders ----
add(40, "CheckpointLoaderSimple", (-500, -600), [CKPT], size=[380, 110])
add(41, "LTXAVTextEncoderLoader", (-500, -450), [TE, CKPT, "default"], size=[380, 130])
add(42, "LTXVAudioVAELoader", (-500, -270), [CKPT], size=[380, 80])

add(43, "LoraLoaderModelOnly", (-500, -150), [DISTILL_LORA, 0.5], size=[380, 110],
    title="蒸馏 LoRA (8步档; 高质量档请旁路)")
link(40, 0, 43, "model")

add(44, "LoraLoaderModelOnly", (-500, 20), [ING_LORA, 1.0], size=[380, 110],
    title="Ingredients IC-LoRA (1.0~1.4)")
link(43, 0, 44, "model")

# ---- prompts ----
add(50, "CLIPTextEncode", (-90, -600), [POSITIVE], size=[520, 420], title="正向: 两段式提示词")
link(41, 0, 50, "clip")
add(51, "CLIPTextEncode", (-90, -150), [NEGATIVE], size=[520, 110], title="负向")
link(41, 0, 51, "clip")

add(52, "LTXVConditioning", (-90, 10), [FPS], size=[300, 100])
link(50, 0, 52, "positive")
link(51, 0, 52, "negative")

# ---- latents + guide ----
add(60, "EmptyLTXVLatentVideo", (-90, 170), [W, H, FRAMES, 1], size=[300, 140])

add(61, "LTXVAddGuide", (280, 170), [0, 1.0], size=[330, 180],
    title="参考图册作为 in-context 引导")
link(52, 0, 61, "positive")
link(52, 1, 61, "negative")
link(40, 2, 61, "vae")
link(60, 0, 61, "latent")
link(30, 0, 61, "image")

add(62, "LTXVEmptyLatentAudio", (280, 400), [FRAMES, FPS, 1], size=[300, 110])
link(42, 0, 62, "audio_vae")

add(63, "LTXVConcatAVLatent", (650, 300), [], size=[280, 70])
link(61, 2, 63, "video_latent")
link(62, 0, 63, "audio_latent")

# ---- sampling ----
add(70, "RandomNoise", (650, -600), [42, "randomize"], size=[280, 100])
add(71, "KSamplerSelect", (650, -470), ["euler_ancestral_cfg_pp"], size=[280, 80])
add(72, "ManualSigmas", (650, -350), [SIGMAS], size=[280, 110], title="8 步蒸馏 sigmas")
add(73, "CFGGuider", (650, -190), [1], size=[280, 120], title="cfg=1 (蒸馏档)")
link(44, 0, 73, "model")
link(61, 0, 73, "positive")
link(61, 1, 73, "negative")

add(74, "SamplerCustomAdvanced", (990, -190), [], size=[280, 130])
link(70, 0, 74, "noise")
link(73, 0, 74, "guider")
link(71, 0, 74, "sampler")
link(72, 0, 74, "sigmas")
link(63, 0, 74, "latent_image")

# ---- decode ----
add(80, "LTXVSeparateAVLatent", (1320, -190), [], size=[280, 80])
link(74, 0, 80, "av_latent")

add(81, "LTXVCropGuides", (1320, -60), [], size=[280, 110], title="剥掉参考 token")
link(61, 0, 81, "positive")
link(61, 1, 81, "negative")
link(80, 0, 81, "latent")

add(82, "VAEDecodeTiled", (1650, -190), [768, 64, 4096, 4], size=[280, 160])
link(81, 2, 82, "samples")
link(40, 2, 82, "vae")

add(83, "LTXVAudioVAEDecode", (1650, 40), [], size=[280, 80])
link(80, 1, 83, "samples")
link(42, 0, 83, "audio_vae")

add(84, "CreateVideo", (1980, -190), [FPS], size=[280, 100])
link(82, 0, 84, "images")
link(83, 0, 84, "audio")

add(85, "SaveVideo", (1980, -20), ["video/LTX23_multiref", "auto", "auto"], size=[380, 400])
link(84, 0, 85, "video")

# ---- notes (frontend-only nodes) ----
add(100, "MarkdownNote", (-2350, -600), [NOTE_MAIN], size=[420, 1250], title="使用说明")
add(101, "MarkdownNote", (-850, 830), [NOTE_SHEET], size=[300, 340], title="图册布局")

wf = {
    "id": "ltx23-multiref-ingredients",
    "revision": 0,
    "last_node_id": max(n["id"] for n in nodes),
    "last_link_id": _lid[0],
    "nodes": nodes,
    "links": links,
    "groups": [
        {"id": 1, "title": "1. 参考图册 Reference Sheet", "bounding": [-1950, -700, 1420, 1650],
         "color": "#3f789e", "font_size": 24, "flags": {}},
        {"id": 2, "title": "2. 模型 + 提示词", "bounding": [-550, -700, 1010, 900],
         "color": "#88A", "font_size": 24, "flags": {}},
        {"id": 3, "title": "3. 采样", "bounding": [620, -700, 690, 1050],
         "color": "#a1309b", "font_size": 24, "flags": {}},
        {"id": 4, "title": "4. 解码输出", "bounding": [1300, -700, 1090, 1050],
         "color": "#3f789e", "font_size": 24, "flags": {}},
    ],
    "config": {},
    "extra": {"ds": {"scale": 0.45, "offset": [2500, 800]}},
    "version": 0.4,
}

# ---------------- validation ----------------
errs = []
for n in nodes:
    t = n["type"]
    if t not in OI:
        if t not in ("MarkdownNote",):
            errs.append(f"unknown node type {t}")
        continue
    spec = OI[t]["input"]
    req = spec.get("required", {})
    opt = spec.get("optional", {})
    linked = {i["name"] for i in n["inputs"]}
    # every required non-widget input must be linked
    for name, v in req.items():
        typ = v[0] if isinstance(v, list) else v
        is_widget = isinstance(typ, list) or typ in (
            "INT", "FLOAT", "STRING", "BOOLEAN", "COMBO",
            "COMFY_DYNAMICCOMBO_V3",
        )
        if not is_widget and name not in linked:
            errs.append(f"{t}#{n['id']} missing required input {name}")
    for name in linked:
        if name not in req and name not in opt:
            errs.append(f"{t}#{n['id']} bogus input {name}")
    # widget count check
    widget_names = [
        name for name, v in req.items()
        if (isinstance(v[0], list) if isinstance(v, list) else False)
        or (isinstance(v, list) and isinstance(v[0], str) and v[0] in
            ("INT", "FLOAT", "STRING", "BOOLEAN", "COMBO", "COMFY_DYNAMICCOMBO_V3"))
    ]
    widget_names = [w for w in widget_names if w not in linked]
    wv = n.get("widgets_values", [])
    if len(wv) < len(widget_names):
        errs.append(f"{t}#{n['id']} widgets_values {len(wv)} < widgets {len(widget_names)} {widget_names}")

seen = set()
for l in links:
    assert l[0] not in seen, "dup link id"
    seen.add(l[0])

if errs:
    print("VALIDATION ERRORS:")
    for e in errs:
        print("  -", e)
    raise SystemExit(1)

json.dump(wf, open(OUT, "w"), ensure_ascii=False, indent=2)
print(f"OK  {len(nodes)} nodes, {len(links)} links -> {OUT}")

# SenseNova U1 多图融合节点(外部 API 调用)

把多图融合("溶图")做成了一个 **ComfyUI 节点** `SenseNovaU1LocalCompose`,
这样外部项目只需向 ComfyUI 发 HTTP 请求即可调用,**无需把模型推理逻辑写进你的 API**。

## 节点

- **node_id**: `SenseNovaU1LocalCompose`
- **显示名**: SenseNova U1 Local Compose (Multi-Ref)
- **位置**: `custom_nodes/ComfyUI-SenseNova-U1/`(fork 自官方,已加此节点)
- **能力**: 多张参考图(角色/道具/场景...)+ prompt(用 `<image>` 占位符按序绑定每张图)
  -> 一张融合图。底层走 `it2i_generate` 多图,图数须 >= `<image>` 占位符数。

### 输入
| 输入 | 说明 |
|---|---|
| `u1_model` | 来自 `SenseNovaU1LocalLoader` 的模型 |
| `image` | 第 1 张参考图(必填) |
| `image2`~`image6` | 第 2~6 张参考图(可选) |
| `prompt` | 提示词,用 `<image>` 按顺序绑定每张图;不写则自动在开头补齐 |
| `auto_size` | true=按第一张图自动定输出尺寸;false=用 width/height |
| `width/height` | 输出尺寸(auto_size=false 时生效) |
| `target_megapixels` | 输出像素预算(默认 1.0=1024²) |
| `input_megapixels` | **每张输入图**像素上限(默认 1.0=1024²)。多图易 OOM,调小此值 |
| `cfg_scale/img_cfg_scale/cfg_norm/timestep_shift/num_steps/seed/think_mode` | 同 edit 节点 |

### 输出
`images`(IMAGE)、`text`、`think_text`、`metadata_json`

## 外部项目怎么调(标准 ComfyUI HTTP API)

ComfyUI 本身就是个 HTTP 服务。你的项目按下面 4 步调用,任何语言都行:

```
1. 上传素材   POST /upload/image          (multipart, 字段名 image)
2. 提交工作流  POST /prompt                {"prompt": <workflow>, "client_id": "..."}
3. 轮询结果   GET  /history/<prompt_id>    直到 status.completed
4. 取图       GET  /view?filename=..&subfolder=..&type=output
```

工作流模板见 `workflows/sense-nova-u1/SenseNova_U1_compose.api.json`
(4 张参考图示例,占位文件名 `CHARACTER_A.png` 等)。调用时:
- 把素材上传后,改对应 `LoadImage` 节点的 `image` 为真实文件名
- 不需要的参考图节点删掉,并去掉 Compose 节点里对应的 `imageN` 输入
- 改 Compose 节点的 `prompt`(占位符数 <= 参考图数)

### Python 最小示例

```python
import json, urllib.request, time

SERVER = "http://你的-comfyui:8188"

# 1. 上传(略,POST /upload/image,字段 image)
# 2. 提交
wf = json.load(open("SenseNova_U1_compose.api.json"))
wf["2"]["inputs"]["image"] = "已上传的角色A.png"   # 替换占位
# ... 其余 LoadImage 同理
data = json.dumps({"prompt": wf}).encode()
r = json.loads(urllib.request.urlopen(
    urllib.request.Request(f"{SERVER}/prompt", data=data,
        headers={"Content-Type": "application/json"})).read())
pid = r["prompt_id"]
# 3. 轮询
while True:
    h = json.loads(urllib.request.urlopen(f"{SERVER}/history/{pid}").read())
    if pid in h and h[pid]["status"].get("completed"): break
    time.sleep(2)
# 4. 取图
for nid, out in h[pid]["outputs"].items():
    for im in out.get("images", []):
        q = urllib.parse.urlencode({"filename": im["filename"],
            "subfolder": im.get("subfolder",""), "type": im.get("type","output")})
        img_bytes = urllib.request.urlopen(f"{SERVER}/view?{q}").read()
        open("result.png","wb").write(img_bytes)
```

> 本仓库的 `scripts/sensenova_compose.py` 是另一种用法(直接加载模型、命令行单次跑),
> 适合本机批处理;而**这个节点 + HTTP API 适合外部服务/别的项目集成**。

## 注意

- ComfyUI 必须用 conda `sensenova` 环境启动(transformers 4.57.1),节点才能加载模型。
- 显存:多图 + 高分辨率易 OOM。L40S 48GB 上,2~4 张图建议 `input_megapixels` 0.6~1.0、
  输出 768~1024。图越多越要调小。
- 节点改动在 fork `o0OooO/ComfyUI-SenseNova-U1`(submodule),已推送。
- 已实测:通过 `POST /prompt` 提交 2 图融合,端到端成功出图。

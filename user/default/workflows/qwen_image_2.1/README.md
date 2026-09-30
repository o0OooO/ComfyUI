# Qwen-Image-2.1（生图 / 编辑 / 多参考）

2026-09-30 新建。**这一代替换掉 `multiref/qwen_image_edit_2511_*`**：同一组参考图、同一 prompt/seed/steps
下，2.1 只用 56s（2511 是 328s），而且三张参考图全部认（2511 把大雁塔那张漏了）。

工作流都是 **API 格式**（`{node_id: {class_type, inputs, _meta}}`），直接 POST 到 `/prompt`：

```bash
bash user/default/scripts/start_comfy.sh        # 必须是 conda sensenova 环境
python /tmp/run_wf.py user/default/workflows/qwen_image_2.1/t2i.api.json
```

| 文件 | 干什么 | 实测 |
|---|---|---|
| `t2i.api.json` | 文生图（中文招牌那种密集文字场景） | 1024² / 25 步 / 145s（含首次加载） |
| `multiref.api.json` | 多参考合成（最多 10 张） | 1024² / 30 步 / **56s** |
| `background_removal.api.json` | 抠图出 RGBA PNG | 原图 1280×1920 / 25 步 / 81s |

## 模型

| 位置 | 文件 |
|---|---|
| `diffusion_models/` | `qwen_image_2.1_int8_convrot.safetensors` |
| `text_encoders/` | `qwen3vl_8b_int8_convrot.safetensors`（`CLIPLoader` type=`qwen_image`） |
| `vae/` | `qwen_image_2.1_vae_bf16.safetensors` |

都在持久盘 `/mnt/models`，重启不丢。

## 三个容易踩的点

**1. `resolution` 不是边长，是总像素预算。**
`TextEncodeQwenImage21.resolution` 填 1024 的意思是「总像素约 1024×1024，宽高比跟参考图，各边对齐到 32 的倍数」。
填 **0 = 保持原图尺寸**（只做 32 对齐）——抠图/局部编辑要的就是 0。

**2. 画布由谁决定，看 latent 接哪。**
- 接 `EmptyLatentImage` → 你自己定画布（`multiref.api.json` 固定 1024² 就是为了和 `cmp_xian3/` 那组横评同画布）。
- 接 `TextEncodeQwenImage21` 的第 3 个输出（`["enc", 2]`）→ 画布跟随 `image_1`，编辑类必须这么接。
- 顺带：`EmptyLatentImage` 是 4 通道而 2.1 要 64 通道，但 `comfy/sample.py:45` 的
  `fix_empty_latent_channels` 会把全零 latent 自动重复到位，所以混用不报错。

**3. `cfg` 必须是 1。**
2.1 的负向走的是 `TextEncodeQwenImage21.negative_prompt`（编码进同一次前向），
不是 KSampler 的双路 CFG。cfg 调大只会糊。

## 多参考怎么写提示词

参考图按 `images.image_1` … `images.image_10` 顺序喂，提示词里用 `<image1>`…`<image10>` 点名。
**编辑类：`image_1` 就是被编辑的那张**，其余是风格/元素来源。
不点名的参考图模型会当背景信息，容易被忽略——想要哪个元素就必须在句子里写出来它来自哪张。

## ⚠ 许可证是退步

Qwen-Image-2.1 是 **Qwen Research 许可（非商用）**。
要商用请用 **Qwen-Image-2512**（Apache-2.0，是目前最新的可商用档）。
老的 Edit-2511 工作流留在 `multiref/` 没删，就是为了这个（另外它现在也跑不起来：
`input/ref_c.png` 不存在）。

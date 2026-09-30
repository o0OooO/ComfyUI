# SenseNova U1.5-8B-MoT（商汤统一多模态生图 / 编辑）

2026-09-30 新建。**这一代不再需要自定义节点、也不再走 conda 里那套 wrapper 推理脚本**——
纯 ComfyUI 核心节点就能跑。老的 `sense-nova-u1/`（U1）留着做对照：同一件编辑任务
U1 那条路 108.1s，U1.5 的 8 步档 **9.8s**。

```bash
bash user/default/scripts/start_comfy.sh
python /tmp/run_wf.py user/default/workflows/sensenova_u1.5/t2i_8step.api.json
```

| 文件 | 配方 | 实测 |
|---|---|---|
| `t2i_base.api.json` | 无 LoRA，50 步 / cfg 4.0 / shift 3.0 | 1024² 67.2s，2048² 187.1s |
| `t2i_8step.api.json` | 8 步蒸馏 LoRA V2，cfg 1.0 | 1024² 9.8s，2048² 12.1s |
| `edit_8step.api.json` | 8 步 + 参考图编辑（中文指令） | 1280×1920 9.8s |

## 模型

| 位置 | 文件 | 说明 |
|---|---|---|
| `checkpoints/` + `diffusion_models/` | `SenseNova-U1.5-8B-MoT-T8-int8-convrot-tagged.safetensors` | 两处都做了软链 |
| `loras/` | `SenseNova-U1.5-8B-MoT-LoRA-8step-V2-comfy.safetensors` | **官方推荐，V2 比 V1 颜色稳、不过锐** |
| `loras/` | `SenseNova-U1.5-8B-MoT-LoRA-8step-ComfyUI.safetensors` | V1，留着对照 |

全在 `/mnt/models`。

## U1.5 是 MoT 统一模型：一个 ckpt 就是全部

`CheckpointLoaderSimple` 的三个输出（MODEL / CLIP / VAE）都从这一个文件出：

- DiT 权重是真的在里面；
- **CLIP 和 VAE 是核心合成的哨兵**（`pixel_space_vae` + `_sensenova_te_sentinel`），
  所以 ckpt 里没有 `vae.` / `text_encoders.` 前缀的键**是正常的**，不要去找缺失的 VAE 文件；
- latent format 是 `HiDreamO1Pixel` —— **latent 本身就是像素** `[B,3,H,W]`，
  所以画布节点是 `EmptyHiDreamO1LatentImage`，不是 `EmptyLatentImage`。

下载来源是社区转换仓 `Milor123/ComfyUI-ConvRot-SenseNova-U1.5-8B-MoT-T8`。
它的 README 说要配 T8mars 的 wrapper 自定义节点 —— **不用**。
核心的两个检测键都在（`fm_modules.vision_model_mot_gen.embeddings.patch_embedding.weight`
和 `language_model.model.layers.0.self_attn.q_proj_mot_gen.weight`），实跑验证过了。

## 采样配方（照抄官方）

```
SenseNovaSamplingOptions(shift=3.0)   ← 官方 --timestep_shift 3.0
KSamplerSelect(euler) + BasicScheduler(normal, steps, 1.0) + SamplerCustom
```

- `normal` + `SenseNovaModelSampling` 出来的 sigma 序列和官方 `upstream_sigmas` 一致，别换 scheduler。
- `resolution_noise_scale` 是**自动的**，不需要额外挂 `ModelNoiseScale` 之类的节点。
- 原生分辨率 2048²；1024² 实测也在分布内。

## 编辑 / 参考图

走**共享的** `HiDreamO1ReferenceImages` 节点（不是 SenseNova 专属节点），
`images.image_1` … `images.image_100`，按序号顺序喂：

```
CLIPTextEncode(正) ─┐
CLIPTextEncode(负) ─┼→ HiDreamO1ReferenceImages → SamplerCustom.positive/negative
LoadImage ──────────┘   (images.image_1)
```

画布要对齐到 32 的倍数，`edit_8step.api.json` 里用
`GetImageSize → ComfyMathExpression "floor(a/32)*32"` 自动算（注意取 **INT 输出，也就是第 2 个口**）。

中文指令跟随得很好，直接写中文，不用翻英文。

## ⚠ V2 LoRA 必须补前缀才能加载

官方的 `SenseNova-U1.5-8B-MoT-LoRA-8step-V2.safetensors` 里的键**没有 `diffusion_model.` 前缀**，
而 ComfyUI 的 `LoraLoader` 靠这个前缀匹配，直接加载会静默不生效（日志里一堆 "not loaded"）。

已经做好的处理：把 882 个键全部加上前缀另存为 `-V2-comfy.safetensors`
（metadata 里记了来源和变换），并且**故意没给原始 V2 文件建软链**，免得 UI 下拉框里选错。
换新版 LoRA 时记得重复这一步。

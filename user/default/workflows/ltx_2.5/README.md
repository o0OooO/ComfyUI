# LTX-2.5（生视频，音画同步）

2026-09-30 新建，同日实测跑通（L40S，权重已在 `/mnt/models`）：

| 工作流 | 输出 | 实测 |
|---|---|---|
| `i2v.api.json` | 704×960 / 121 帧 / 24fps / 带音轨 | 321s（含首次加载权重） |
| `multiref_ingredients.api.json` | 512×896 / 121 帧 / 24fps / 带音轨 | 100s |

## 0. 权重（换机器时重来一遍）

```
# 1) 浏览器各点一次 "Agree and Access"
#    https://huggingface.co/Lightricks/LTX-2.5
#    https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients
# 2) 本机登录
hf auth login
# 3) 下载（约 41G，全部落持久盘 /mnt/models）
bash user/default/scripts/fetch_ltx25.sh
```

`fetch_ltx25.sh` 会先体检再下载：没 token、或者条款没点，它会直接告诉你差哪一步
（gated 仓库的元数据是公开的，**元数据 200 不代表能下文件**，所以脚本是拿单个文件试 range GET 来判断的）。
下完要让 ComfyUI 重扫模型目录（重启或 UI 里 Refresh），否则 `/prompt` 会报 value not in list。

子命令：`base` / `iclora` / `enhancer`（可选的提示词增强器 5.2G，不 gated）/ `check`（只体检）。

## 1. 工作流

| 文件 | 干什么 |
|---|---|
| `i2v.api.json` | 图生视频，两阶段（低分辨率 8 步 → latent 2× 上采样 → 3 步精修），带同步音频 |
| `multiref_ingredients.api.json` | 多参考，拼图 reference sheet + Ingredients IC-LoRA，单阶段 |

```bash
python /tmp/run_wf.py user/default/workflows/ltx_2.5/i2v.api.json
```

## 2. 相对 LTX-2.3 变了什么

| | 2.3 | 2.5 |
|---|---|---|
| 权重形态 | 一个 all-in-one checkpoint | 拆成 4 类独立文件 |
| 加载节点 | `CheckpointLoaderSimple` + `LTXAVTextEncoderLoader` + `LTXVAudioVAELoader` | `UNETLoader` + `CLIPLoader(type=ltxv)` + `VAELoader` ×2 |
| 文本编码器 | Gemma-3-12B | **Gemma-4-12B（带投影层，2.5 专用）** |
| Guider | `CFGGuider` | `LTXVDualCFGGuider`（video_cfg / audio_cfg 分开） |
| 采样器 | `euler_ancestral_cfg_pp` | `euler_ancestral`（cfg=1 时 cfg_pp 没意义） |
| IC-LoRA | 直接接 `LTXVAddGuide` | 多一个 **`GetICLoRAParameters`** 节点（见下） |

所以不能沿用 `restore_ltx23_models.sh`，也别把 2.3 的工作流改改文件名就用。
2.3 那套（`video_ltx2_3_i2v.json` / `video_ltx2_3_multiref_ingredients.json`）原样留着。

## 3. 两条硬约束（踩了就静默变糊 / 直接抛异常）

**① 分辨率：两阶段流程里目标宽高必须能被 64 整除。**
`EmptyLTXVLatentVideo` 是 `height // 32`（整除，不够就**静默截断**）。
第一阶段跑的是目标尺寸的一半，所以「一半还要能被 32 整除」= 目标能被 64 整除。
官方模板用 `ResolutionSelector(16:9, 0.9MP, 32)` 出 1280×736 —— 736/2=368 不是 32 的倍数，
实际输出会变成 1280×704。本工作流直接写死数字（默认 704×960 竖版）避开这个坑。

参考官方分辨率表（multiple=32）：0.3MP→736×416，0.5→960×544，0.9→1280×736，1.2→1504×832，2.0→1920×1088。

**② 帧数必须是 8n+1**，并且视频和音频要一致：
`duration × fps + 1`，默认 5×24+1 = **121**。
`EmptyLTXVLatentVideo.length`、`LTXVEmptyLatentAudio.frames_number`、
（多参考里还有）`RepeatImageBatch.amount` 三处必须同步改，否则音画不同步或参考帧对不上。

## 4. i2v 的两阶段是怎么回事

```
首帧 ─ ResizeImageMaskNode(长边1536) ─ LTXVPreprocess(18)
                                          │
   EmptyLTXVLatentVideo(半分辨率,121) ─ LTXVImgToVideoInplace(strength 0.7)
                                          ├─ LTXVConcatAVLatent ← LTXVEmptyLatentAudio
                                          └─ SamplerCustomAdvanced(8 步 ManualSigmas)
                                               ↓ LTXVSeparateAVLatent
                            LTXVLatentUpsampler(2×) ─ LTXVImgToVideoInplace(strength 1.0)
                                          ├─ LTXVConcatAVLatent ← 上一阶段的音频 latent
                                          └─ SamplerCustomAdvanced(3 步 0.85→0)
                                               ↓ LTXVSeparateAVLatent
                        VAEDecodeTiled + LTXVAudioVAEDecode → CreateVideo → SaveVideo
```

几个官方细节，别自己"优化"掉：

- **`LTXVPreprocess(img_compression=18)`**：给首帧压一遍 JPEG。不压的话第一帧过锐，和后续帧质感断层。
- **第一阶段 `strength=0.7`，第二阶段 `strength=1.0`**。第一阶段留 0.3 给模型重画首帧，
  否则运动起不来；第二阶段才把首帧硬锁成输入图。
- **8 步 sigmas `1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0`**：
  前 5 步几乎不降噪，这是 2.5 的 pixel-diffusion 关键帧阶段，照抄别改。
- **精修 sigmas 从 0.85 起**（不是 1.0），所以构图由第一阶段定，第二阶段只加细节。
  第二阶段种子官方固定 42，换它只影响细节。
- **蒸馏版 cfg = 1/1**。要换 dev 基座（`ltx-2.5-22b-dev-transformer-*`）就得改成 3/7 + 30 步线性 sigmas。
- 音频不是后配的：`LTXVConcatAVLatent` 把音视频拼成一个 AV latent 一起去噪，
  所以提示词里**要写环境音**（"Ambient audio: ..."）。

官方模板里还有一路可选的提示词增强器（`TextGenerateLTX2Prompt` + `ComfySwitchNode`，默认关），
需要另外 5.2G 的 `gemma4_e2b_it_int8_convrot`，会多花 1-2 分钟。本工作流没接，要用就
`bash user/default/scripts/fetch_ltx25.sh enhancer` 再自己加。

## 5. 多参考：还是"一张拼图"，不是喂多张图

和 2.3 的思路一致（见 `user/default/docs/LTX23_MULTIREF_GUIDE.md`）：
4 张参考 → `ResizeAndPadImage` 成等大格子 → `ImageStitch` 拼成 2×2 图册 →
`RepeatImageBatch` loop 成 121 帧静态视频 → `LTXVAddGuide` 当 in-context 条件 →
采样后 `LTXVCropGuides` 把参考 token 剥掉。

默认布局（竖版 512×896，格子 256×448）：

```
+-----------+-----------+
| 主角 A    | 主角 B    |
+-----------+-----------+
| 关键道具  | 场景      |
+-----------+-----------+
```

2.5 这边多了三件事：

**① 必须接 `GetICLoRAParameters`。**
它从 IC-LoRA 的 safetensors metadata 里读 `reference_downscale_factor` 喂给 `LTXVAddGuide`
（`comfy_extras/nodes_lt.py:22`）。2.3 时代没这个节点；漏了会按 factor=1 处理，参考图的 token 数不对。

**② latent 的宽高（= 像素 /32）必须能被这个 factor 整除**，否则 `LTXVAddGuide` 直接抛
`Latent spatial size ... must be divisible by reference_downscale_factor`。
默认 512×896 → latent 16×28，能被 2 和 4 整除。
反例：16:9 的 736×416 → latent 23×13 全是奇数，factor=2 就炸。

**③ 图册的宽高比必须和输出视频一致。**
`LTXVAddGuide.encode()` 内部是 `common_upscale(..., crop="center")`，
比例不对会把格子裁掉。改横版就是 `EmptyLTXVLatentVideo` 改 896×512 + 4 个格子改 448×256。

提示词必须两段式（和训练时一致）：

- `### Reference Sheet Description` —— 逐格描述图册里有什么（外观）
- `### Target Description` —— 描述要生成的动作/镜头（叙事）

Target 段里要**用图册里同样的措辞**把角色和道具再写一遍，模型才会把它们绑到参考上。
**⚠ 默认 prompt 是占位模板（`<describe ...>`），`ref_a/ref_b.png` 也只是色块测试卡。**
原样跑出来的视频就是一段 2×2 图册本身（音频也是静音）——不是图坏了，是没有 Target。
先换成真实参考图、把两段都写实再跑。

实测经验（真人 ×2 + 三彩马 + 大雁塔）：人物外观、服装、道具都绑上了，
但**场景被人物参考图自带的背景（麦田、皮卡）盖过了**，大雁塔没出来，首帧还闪过一个重复的人。
人物格尽量用干净/纯色背景的图，别用带完整场景的剧照；两格都是同一人时也别让两张背景不同。

角色漂移就把 IC-LoRA 强度提到 1.2~1.4；跟得太死、画面不动就把 `LTXVAddGuide.strength` 降到 0.8~0.9。

## 6. 仓库里还有什么可选文件

| 文件 | 用途 |
|---|---|
| `ltx-2.5-22b-dev-transformer-comfy-int8-convrot` (21.5G) | 非蒸馏基座，质量更高，要 cfg 3/7 + 30 步 |
| `ltx-2.5-22b-distilled-lora-450-bf16` (8.9G) | 给 dev 基座加速用的蒸馏 LoRA |
| `ltx-2.5-22b-distilled-transformer-nvfp4` (18.7G) | ⚠ 别用：L40S 是 sm_89，NVFP4 只能模拟 |
| `ltx-2.5-latent-temporal-upscaler-x2` (0.26G) | 时间轴 2× 插帧（本工作流没用） |
| `ltx-2.5-duration-head-bf16` (小) | Auto Duration：从动作描述预测该多长 |
| `ltx-2.5-video-vae-conv-bf16` (1.45G) | VAE 的 conv 变体 |

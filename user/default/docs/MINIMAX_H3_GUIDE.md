# MiniMax H3 本地部署（g6e.2xlarge / L40S 48GB）

MiniMax H3 是 omni-modal 生成模型，**视频和 32kHz 立体声音频在同一次前向里联合生成**，
不是后期配音。2026-07-28 开源，ComfyUI 主线原生支持（无需第三方节点）。

上游支持提交：
- `57500fc5` (2026-08-03) `feat: Support MiniMax-H3 (CORE-375) #15224`
- `16e3f303` (2026-08-03) H3 VAE 设备 cast 修复

## 已验证配置

本机 L40S 48GB 跑的是 int8_convrot 量化组合（**不是** 官方模板推荐的 NVFP4）：

| 组件 | 文件 | 大小 |
|---|---|---|
| diffusion_model | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | 21 GB |
| text_encoder | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | 27 GB |
| video VAE | `minimax_h3_video_vae_fp16.safetensors` | 5.2 GB |
| audio VAE | `minimax_h3_audio_vae_fp32.safetensors` | 0.6 GB |

权重在 `/mnt/models/`（EBS 持久盘），`models/` 下是逐文件软链 —— 见 [[ebs-persistent-model-disk]] 约定。

### 为什么用 int8 而不是官方模板的 NVFP4

L40S 是 Ada（sm_89）。ComfyUI 的 `supports_nvfp4_compute()` 要求 `props.major >= 10`
（Blackwell sm_100+），否则走模拟路径。启动日志确认：

```
Native ops: float8_e4m3fn, float8_e5m2, int8_tensorwise, convrot_w4a4 , emulated ops: nvfp4, mxfp8
```

`int8_tensorwise` + `convrot_w4a4` 有原生算子，NVFP4 只能模拟。所以本机选 int8 版 TE。
换到 B200 之类 Blackwell 卡再用 `qwen3vl_32b_minimax_h3_nvfp4_awq`（15.7GB，更省显存）。

## 启动

```bash
python main.py --listen 0.0.0.0 --port 8188 --reserve-vram 1.5
```

47GB 权重装不进 48GB 卡，靠 ComfyUI 分阶段换入换出（TE 编码完就卸载）。
峰值显存 36.3GB，系统内存用到 42/61GB —— RAM 是紧的，别同时跑别的模型。

## 跑一条

```bash
python user/default/scripts/gen_minimax_h3_t2va.py
```

## 实测性能（1344×768，124 帧 ≈ 5.2s，30 步）

- 模型初始化：2m21s
- 采样：~34s/step
- **端到端：22m07s**

慢的原因：33B dense transformer，且**首个开源版只有 full attention**
（官方的 sparse attention 未随开源发布，说是后续更新）。

输出验证：`1344x768 @ 24fps h264` + `32000Hz stereo aac`，两条流都是 5.167s，
音频 mean_volume -17.9 dB（有真实内容，非静音）。

## 关键约束

- **length 必须落在 17k+5 网格上**：124 / 141 / 158 …（`align_frame_count()` 会向上取整）。
  124 帧 = 5.17s。训练范围约 124–362 帧。
- **画布是 768 短边**，面积上限 768×1344，每轴对齐 32。1344×768 是原生尺寸。
- **H3-Context-IR 没开源**。官方明确说它"对最终输出质量至关重要" —— 是个托管的多阶段
  prompt 预处理系统（指令解析、跨模态关联、时序理解）。自建只有 H3-Base，得按
  `docs/VIDEO_PROMPT_WRITING_GUIDE_*.md` 自己写 prompt，或付费调 API
  （$0.90/M 输入 tokens，$3.60/M 输出）。
- prompt 要**把画面和音频写在同一段里**（对白、音效、音乐），模型联合建模。
- 许可证是 `minimax-h3-community-license-agreement`，不是 Apache。商用前看
  `docs/QA-about-License.md`。

## 节点

| 节点 | 用途 |
|---|---|
| `MiniMaxH3ImageToVideo` | t2va（不接图）+ fl2va（接 first_frame / last_frame） |
| `MiniMaxH3ReferenceToVideo` | ref2va：≤9 图 / ≤3 视频 / ≤3 音频，总计 ≤12 个文件 |
| `EmptyMiniMaxH3LatentAV` | 空的联合 AV latent |
| `MiniMaxH3SigmaShift` | shift_video=12.0 / shift_audio=3.0（默认值即官方推荐） |

ref2va 的 prompt 里用 `<Picture i>` / `<Video k>` / `<Audio j>` 引用素材，序号按类型 1-based。
注意 `ref_image_size="max"`（2048 短边）identity 保真更好但慢数倍 —— 参考 token 每步都参与计算。

采样出来的 latent 是 NestedTensor pair（video + audio），两个 VAE 各取自己那条流：
`VAEDecode` 拿视频，`VAEDecodeAudio` 拿音频，再用 `CreateVideo` 合成。

## 全量 bf16 要什么配置

全量一套 ≈ 124GB 权重（transformer 66GB + Qwen3-VL-32B TE 51.5GB + VAE 5.8GB），
算上激活和 2K regenerate 需要 160–200GB 显存。

**但 ComfyUI 没有针对 H3 的 tensor parallel**，多卡实例的显存加不起来（g6e.12xlarge 是
4×48GB，单卡还是 48GB）。真要跑全量 bf16 得用官方 diffusers pipeline，不走 ComfyUI。

ComfyUI 内想提质量，务实选择是 g6e.16xlarge（$7.58/h）：单卡仍 48GB，但 512GB 系统内存
能让 offload 从容得多，可跑 `pruned_bf16`（40GB）而非 int8。

## 自建 vs 官方 API

API 按秒计费：768P **$0.08/s**，2K **$0.13/s**，768P→2K regenerate $0.05/s。
输入音频免费，图片前 5 张免费之后 $0.04/张。

5 秒 768P 一条 = $0.40。本机 $2.24/h ÷ $0.40 ≈ 一小时机器钱抵 5.6 条；
而本机 22 分钟才出一条 —— **纯出片成本 API 便宜得多**。
自建的价值在可微调、无内容审核、无速率限制、离线可控。

视频资源包（$1000 起）目前不支持 H3，只能 pay-as-you-go。

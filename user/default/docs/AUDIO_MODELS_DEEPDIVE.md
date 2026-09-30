# 音频生成模型深度调研（音乐 / 音效）

> 调研日期：2026-06-09 ｜ 数据来源：HuggingFace 模型页 + 各官方仓库 README
> 选型范围：音乐 = ACE-Step 1.5 + HeartMuLa；音效 = stable-audio-3-small-sfx
> TTS 部分（higgs-audio-v3）经评估后暂缓本地落地，原因见文末「TTS 备注」。

---

## 一览对比表

| 维度 | ACE-Step 1.5 | HeartMuLa-oss-3B | stable-audio-3-small-sfx |
|------|--------------|------------------|--------------------------|
| 类别 | 音乐生成 | 音乐生成（整首带唱） | 音效 SFX |
| 厂商 | ACE Studio / ACE-Step 团队 | HeartMuLa Team | Stability AI |
| 开源时间 | 2026-01-23 | 2026-01-14 | 2026-05-17 |
| 参数量 | ~3.5B（LM+DiT 混合） | ~3.9B | 567.6M |
| 下载量 | 211.7K | 18.2K | 8.3K（Gated 门控） |
| Likes | 766 | 257 | 49 |
| 许可证 | **MIT（可商用）** | **Apache-2.0（可商用）** | other（**商用受限**，需查条款） |
| ComfyUI 支持 | ✅ **原生核心节点** | ✅ 第三方自定义节点 | ✅ **原生**（Comfy-Org 打包单文件） |
| 显存需求 | **< 4GB**（消费级可跑） | 中等，支持多卡拆分/懒加载 | 低（567M，最省） |
| 推理速度 | A100 <2s / 3090 <10s 整曲 | RTF ≈ 1.0（实时级） | 快（模型小） |

---

## 1. 音乐生成

### 1.1 ACE-Step / Ace-Step1.5

- **链接**：https://hf.co/ACE-Step/Ace-Step1.5
- **厂商**：ACE-Step 团队（ACE Studio 背景）
- **开源时间**：2026-01-23（页面更新 2026-02-03）
- **下载 / Likes**：211.7K ↓ / 766 ❤（社区热度最高的开源 text2music）
- **架构**：混合架构 = LM 规划器（planner）+ Diffusion Transformer（DiT）合成器
- **许可证**：MIT，训练数据为授权/免版税/合成数据，**可商用、零版税**

**特点**
- 三档模型：`turbo`(8 步最快) / `sft`(50 步高质) / `base`(50 步最灵活)
- 能力：text-to-music、翻唱(cover)、重绘(repainting)、人声转 BGM(vocal-to-BGM)
- 50+ 语言，prompt 遵循度好

**优势**
- 🟢 **可商用 MIT**，无版税顾虑
- 🟢 **极省显存（<4GB）**，消费级显卡可跑
- 🟢 **极快**：A100 <2s、RTX 3090 <10s 出整曲
- 🟢 **ComfyUI 原生支持**，生态最好（56+ Demo Space）
- 🟢 功能全面（翻唱/重绘/编辑）

**劣势**
- 🔴 人声演唱 / 歌词贴合能力弱于 HeartMuLa、SongGeneration，更偏纯音乐与伴奏
- 🔴 复杂歌曲结构（主歌/副歌切换）表现一般

**推理代码**
```python
from transformers import pipeline
pipe = pipeline("text-to-audio", model="ACE-Step/Ace-Step1.5", trust_remote_code=True)
```

---

### 1.2 HeartMuLa-oss-3B

- **链接**：https://hf.co/HeartMuLa/HeartMuLa-oss-3B ｜ GitHub: https://github.com/HeartMuLa/heartlib
- **厂商**：HeartMuLa Team
- **开源时间**：2026-01-14
- **下载 / Likes**：18.2K ↓ / 257 ❤（口碑好但热度已过峰，被低估）
- **架构**：heartmula（~3.9B），含配套 `HeartCodec` 与 `HeartTranscriptor`（whisper 系歌词转写）
- **许可证**：Apache-2.0，**可商用**
- **论文**：arXiv:2601.10547 ｜ Demo: https://heartmula.github.io/

**特点**
- **整首带人声歌曲生成**（conditioned on lyrics + tags），歌词可控性强
- 多语言歌词支持（覆盖几乎所有语言）
- 最长 **240 秒（4 分钟）**，可通过 `--max_audio_length_ms` 调整
- 支持多卡拆分：`--mula_device cuda:0 --codec_device cuda:1`，单卡可用懒加载省显存
- 同系列：`HeartMuLa-oss-3B-happy-new-year`(节日微调)、`HeartMuLaGen`、`HeartTranscriptor-oss`、`...-bf16`(省显存版)

**优势**
- 🟢 **整首带唱 + 歌词条件控制**，是该能力的开源代表之一
- 🟢 **Apache-2.0 可商用 + 多语言**组合稀缺
- 🟢 配套歌词转写模型，生态完整
- 🟢 多卡/懒加载部署灵活

**劣势**
- 🔴 RTF ≈ 1.0（实时级，但比 ACE-Step turbo 慢很多）
- 🔴 4B 体量显存需求高于 ACE-Step
- 🔴 当下热度回落、更新放缓
- 🔴 README 信息较简略，需翻 GitHub

**推理代码**
```bash
git clone https://github.com/HeartMuLa/heartlib && cd heartlib && pip install -e .
# 下载 ckpt 后：
python ./examples/run_music_generation.py --model_path=./ckpt --version="3B"
```

> **选型建议（音乐）**：要快、省显存、可商用、功能全 → **ACE-Step 1.5**；要整首带唱 + 歌词控制 + 可商用 → **HeartMuLa**。两者互补，可同时部署。

---

## 2. 音效生成

### 2.1 stabilityai/stable-audio-3-small-sfx

- **链接**：https://hf.co/stabilityai/stable-audio-3-small-sfx
- **厂商**：Stability AI
- **开源时间**：2026-05-17
- **下载 / Likes**：8.3K ↓ / 49 ❤（**Gated 门控模型，需在 HF 同意条款后下载**）
- **参数量**：567.6M（四个选型里最小最省）
- **架构**：diffusion，基于 `stable-audio-3-small-sfx-base` 微调
- **许可证**：other（**商用受限**，需逐条核对 Stability 协议）
- **语言**：英文 prompt

**特点 / 优势**
- 🟢 **专精音效 SFX**，prompt 控制精准，最新一代质量高
- 🟢 模型小（567M），**显存需求最低、推理快**
- 🟢 同系列还有 `-small-music`(音乐)、`-medium`(音乐+音效通用)

**劣势**
- 🔴 **许可证非完全开放**，商用前必须确认条款
- 🔴 Gated，需申请/同意才能下载
- 🔴 偏短片段，仅英文 prompt
- 🔴 ComfyUI **暂无 v3 原生支持**（仅 open-1.0 原生）

> **替代/参考**：若要可商用音效，可考虑 `OpenMOSS-Team/MOSS-SoundEffect-v2.0`(Apache-2.0)；要成熟生态可用 `stabilityai/stable-audio-open-1.0`（ComfyUI 原生支持）。

---

## 3. ComfyUI 接入方案汇总

| 模型 | ComfyUI 现状 | 接入方式 |
|------|--------------|----------|
| **ACE-Step 1.5** | ✅ 原生核心节点（含专用 `*1.5` 节点） | 直接用官方节点 + Comfy-Org 打包权重 |
| **stable-audio-3-sfx** | ✅ 原生（核心 `StableAudio3` 模型类） | Comfy-Org 打包单文件，下权重即用 |
| **HeartMuLa** | ✅ 第三方节点 | `git clone` 安装社区节点 |

> 本机已下好权重并建软链(临时盘),恢复脚本见 `scripts/restore_audio_comfy_models.sh`,
> 存储说明见 [NVME_STORAGE_NOTES.md](./NVME_STORAGE_NOTES.md)。

### 3.1 ACE-Step 1.5 —— ComfyUI 原生（推荐路径）
本机 ComfyUI 核心已内置 ACE-Step 1.5 专用节点（`comfy_extras/nodes_ace.py` + `comfy/ldm/ace/ace_step15.py`）：
- `Load Checkpoint` — 加载 `ace_step_1.5_turbo_aio.safetensors`（all-in-one,含 VAE+文本编码器）
- `TextEncodeAceStepAudio1.5` — tags + lyrics + bpm/duration/调式/语言等
- `EmptyAceStep1.5LatentAudio` — 设定时长 latent
- `ModelSamplingAuraFlow` → `KSampler` → `VAEDecodeAudio` → `SaveAudioMP3`
- `ReferenceTimbreAudio`（实验性）— 可给音色参考

> 官方模板:ComfyUI 启动后「模板 → 音频」里有 `audio_ace_step_1_5_checkpoint`，checkpoint 默认就指向 `ace_step_1.5_turbo_aio.safetensors`，开箱即用。

### 3.2 stable-audio-3-sfx —— ComfyUI 原生
本机核心已支持 Stable Audio 3（`comfy/supported_models.py` 的 `StableAudio3` 类）：
- `Load Checkpoint` — `stable_audio_3_small_sfx.safetensors`
- 文本编码器 — `text_encoders/t5gemma_b_b_ul2.safetensors`
- 标准音频流程出音效 → `SaveAudio`

### 3.3 HeartMuLa —— 社区自定义节点
- 节点仓库：`https://github.com/benjiyaya/HeartMuLa_ComfyUI`
- 安装：`cd ComfyUI/custom_nodes && git clone https://github.com/benjiyaya/HeartMuLa_ComfyUI && pip install -r requirements.txt`
- 依赖 `torchtune`/`torchao`/`soundfile`（不锁 transformers 版本，与 sensenova 环境冲突风险小）
- 模型放到对应目录后，节点暴露 lyrics/tags/时长/多卡设备等参数

---

## 4. 选型结论

| 用途 | 首选 | 理由 | 商用注意 |
|------|------|------|----------|
| 纯音乐 / 伴奏 / 翻唱 | **ACE-Step 1.5** | 快、省显存、MIT、ComfyUI 原生 | ✅ 可商用 |
| 整首带唱 + 歌词控制 | **HeartMuLa** | 演唱质量 + 歌词控制 + Apache | ✅ 可商用 |
| 音效 SFX | **stable-audio-3-sfx** | 专精、质量高、最省显存、原生 | ⚠️ 商用查条款，或换 MOSS-SFX(Apache) |

**ComfyUI 落地优先级**：ACE-Step 1.5 与 stable-audio-3-sfx（均原生，下权重即用）> HeartMuLa（装社区节点）。

---

## TTS 备注（higgs-audio-v3，暂缓）

TTS 方向曾评估 `bosonai/higgs-audio-v3-tts-4b`（102 语言、零样本克隆、细粒度情感控制，表现力顶级），但**暂缓本地落地**，原因：
- **架构未进主线 transformers**：v3 的 `higgs_multimodal_qwen3` 在本机 transformers 4.57 里无对应类，且仓库无 `auto_map`/建模代码、无 vocoder，**纯 transformers 节点跑不起来**。
- **唯一本地路径是 sglang-omni 服务**（需单独起服务 + ≥16–24GB 显存,建议 g5.xlarge A10G 单独机器），与现有 ComfyUI 抢显存。
- **非商用许可**：商用需单独授权。

> 若后续要做 TTS：可商用替代 `Qwen3-TTS`(Apache,中文强)、`hexgrad/Kokoro-82M`(Apache,轻量)、`k2-fsa/OmniVoice`(多语言克隆)；
> 要 higgs v3 则走 g5 + sglang-omni + ComfyUI HTTP 节点的解耦方案。

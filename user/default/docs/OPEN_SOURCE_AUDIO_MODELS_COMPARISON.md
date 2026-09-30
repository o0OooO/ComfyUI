# 开源音频生成模型对比(2024–2026)

> 数据来源:HuggingFace 模型页 + model card(一手核对)。标 ⏳ 的 benchmark 数字未独立核实,仅作方向参考——选型主要看场景和优劣势那两栏。
> 整理日期:2026-06-05

---

## 一、音乐生成

| 模型 | 许可证 | 参数 | 速度 / 显存 | 最擅长场景 | 优势 | 劣势 / 坑 |
|---|---|---|---|---|---|---|
| **ACE-Step v1-3.5B** | **Apache 2.0** ✅商用 | 3.5B | A100 **20s 出 4 分钟**(RTF 27×);4090 RTF 34×;3090 12.8× | 批量出整曲、要快、要商用 | 真商用;快到能交互;支持词→歌 | 惊艳度不如 Suno;细节质感一般 |
| **MusicGen-medium** | CC-BY-NC(非商用) | 1.5B | 自回归,较慢 | 研究、demo、可控生成 | 生态最成熟、ComfyUI 节点多;melody 版可跟旋律 | 非商用;慢;单次约 30s,长曲要拼 |
| **MusicGen-large** | CC-BY-NC | 3.3B | 更慢 | 追求 MusicGen 最高质量 | 质量比 medium 好 | 非商用;慢、吃显存 |
| **YuE-s1-7B** | **Apache 2.0** ✅ | 6B | 慢;量化后约 8GB 显存 | **带人声的完整歌曲**(主唱+伴奏) | 唯一开源能直接出人声歌曲 | 很吃显存/时间;不适合实时 |
| **Stable Audio Open 1.0** | Stability Community(商用需授权) | 1B | 最长 47s,44.1kHz 立体声 | 短音乐片段、loop、高保真 | 采样率高;音乐+音效两用 | 商用要单独授权;做不了长曲 |

**决策**:商用配乐→**ACE-Step** | 研究可控→**MusicGen** | 人声歌曲→**YuE** | 高保真短片段→**Stable Audio Open**

---

## 二、音效 / Foley / 环境音

| 模型 | 许可证 | 参数 | 最擅长场景 | 优势 | 劣势 |
|---|---|---|---|---|---|
| **AudioGen-medium** | CC-BY-NC(非商用) | 1.5B | 文本→环境音/拟音("狗叫""下雨") | Meta 出品,音效基线 SOTA;短音效稳 | 非商用;只做短音效,不做音乐 |
| **Stable Audio Open 1.0** | Stability Community | 1B | 音效 + 短音乐通吃 | 一模型两用;44.1kHz | 商用需授权 |
| **Tango 2 / AudioLDM2** | 多为研究许可 | — | 文本→音效,扩散路线 | prompt 遵循度好 | 研究许可;速度一般 |

**决策**:纯音效→**AudioGen**(质量基线)或 **Stable Audio Open**(顺带能做音乐)。Foley 开源整体不如音乐成熟。

---

## 三、TTS / 歌声合成

| 模型 | 许可证 | 参数 | 关键指标 | 速度 | 最擅长场景 | 优势 | 劣势 |
|---|---|---|---|---|---|---|---|
| **Kokoro-82M** | **Apache 2.0** ✅ | 82M | TTS Arena 高排名 ⏳ | 极快,CPU 可跑 | 海量播报、有声书、边缘设备 | 极小极快极省;真商用 | 不能克隆;音色固定;情感一般 |
| **CosyVoice2-0.5B** | **Apache 2.0** ✅ | 0.5B | 中 CER 1.45%/SIM 75.7%;英 WER 2.57%/SIM 65.9% | 流式 **150ms** | 实时对话、多语言+方言+克隆 | 低延迟流式;18+ 中文方言;商用 | 比 Kokoro 重;部署稍复杂 |
| **F5-TTS** | CC-BY-NC(非商用) | ~0.3B | WER/SIM ⏳ | 流匹配,较快 | 零样本声音克隆 | 克隆相似度口碑强 | 非商用;长文本偶不稳 |
| **XTTS-v2** | Coqui(受限) | — | — | 中等 | 多语言克隆、社区方案 | 17 语言;教程多 | Coqui 已停运;许可受限;新项目不推荐 |
| **DiffSinger** | 见仓库 | — | — | 需乐谱预处理 | **歌声合成(SVS)** | 专门做唱歌 | 要乐谱+音素标注,门槛高 |

**决策**:海量省钱商用→**Kokoro** | 实时+中文方言+克隆→**CosyVoice2** | 最佳克隆(非商用)→**F5-TTS** | 唱歌→**DiffSinger**

---

## 跨场景速查

| 需求 | 选 |
|---|---|
| 商用背景音乐、要快 | **ACE-Step** |
| 带人声歌曲 | **YuE** |
| 音效 / 拟音 | **AudioGen** 或 **Stable Audio Open** |
| 海量 TTS、省成本、商用 | **Kokoro-82M** |
| 实时语音 + 中文方言 + 克隆 | **CosyVoice2** |
| 最佳声音克隆 | **F5-TTS** |
| 唱歌 / 歌声合成 | **DiffSinger** |

---

## 商用许可证速记

- **可商用(Apache 2.0)**:ACE-Step、YuE、Kokoro、CosyVoice2
- **非商用(CC-BY-NC)**:MusicGen、AudioGen、F5-TTS
- **需单独授权**:Stable Audio 系列

---

## HuggingFace 月下载量(人气参考)

| 模型 | 月下载 |
|---|---|
| Kokoro-82M | 13.8M(TTS 最高) |
| XTTS-v2 | 10M |
| MusicGen-medium | 1.5M(音乐最高) |
| F5-TTS | 677k |
| CosyVoice2 系列 | ~150k |
| ACE-Step | 49k(音乐新秀人气最高) |
| Stable Audio Open 1.0 | 44.5k |

---

## 数据诚实声明

- license、参数、ACE-Step/CosyVoice2 的速度与指标是从 model card 一手核对的。
- 标 ⏳ 的(各家 FAD/CLAP/WER 横向对比)未独立核实,未编造数字。
- 如需某模型的精确论文 benchmark,可单独核查其 arXiv 论文。

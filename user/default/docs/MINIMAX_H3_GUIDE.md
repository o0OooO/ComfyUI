# MiniMax H3 本地部署（g6e.2xlarge / L40S 48GB）

MiniMax H3 是 omni-modal 生成模型，**视频和 32kHz 立体声音频在同一次前向里联合生成**，
不是后期配音。2026-07-28 开源，ComfyUI 主线原生支持（无需第三方节点）。

上游支持提交：
- `57500fc5` (2026-08-03) `feat: Support MiniMax-H3 (CORE-375) #15224`
- `16e3f303` (2026-08-03) H3 VAE 设备 cast 修复

2026-08-14 合的一批 H3 相关上游改动（`19c87937`，跟着 66 个 commit 一起进来）：
- `2a68ce33` Optimize MiniMax-H3 VAE
- `62b3c94b` Fix peak memory issue with H3
- `bf4c9a08` Implement comfy kitchen attention
- `bdcb886a` Fix sampler issues for audio with minimax, support more samplers
- `344b4398` Support asym w4a8_int ← w4a8 量化要它
- `bbda8364` Support int8_convrot VAE
- `ddbaa875` minimax: early detect qkv vs q,k,v
- `efd4e951` Minimax Music 3 + CUDA Graphs 核心支持
- `e01fb4c5` MiniMaxH3AddGuide（任意帧锚定 image/audio guide，ref2va 可用）

配套：`comfy-kitchen` 0.2.26→0.2.31，装在 `sensenova` env（**server 跑的是这个 env，
不是 `test`**）。

## 已验证配置

本机 L40S 48GB 跑的是 int8_convrot 量化组合（**不是** 官方模板推荐的 NVFP4）：

| 组件 | 文件 | 大小 |
|---|---|---|
| diffusion_model | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | 21 GB |
| text_encoder | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | 27 GB |
| video VAE | `minimax_h3_video_vae_fp16.safetensors` | 5.2 GB |
| audio VAE | `minimax_h3_audio_vae_fp32.safetensors` | 0.6 GB |

第二套量化：`AX1Y2JP/MiniMax-H3-W4A8-ConvRot` 的 `*_pruned_w4a8_mixed`（transformer
12.5GB + TE 16.5GB），fl2va / ref2va 都有。取舍见下面的性能表 —— 冷启快、显存省，
但每步慢 8%。

蒸馏 LoRA：`lightx2v/Minimax-h3-Turbo` 三个 comfyui 版在
`/mnt/models/loras/minimax_h3_turbo/`。

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
python user/default/scripts/gen_minimax_h3_t2va.py                  # 定稿:8 步 turbo + int8
python user/default/scripts/gen_minimax_h3_t2va.py --preset 4step   # 预览用,浅景深会塌
python user/default/scripts/gen_minimax_h3_t2va.py --preset base    # 回到 30 步基线
python user/default/scripts/gen_minimax_h3_t2va.py --quant w4a8     # 换小量化
python user/default/scripts/gen_minimax_h3_t2va.py --write-only     # 只重生成工作流 JSON
```

`--steps` / `--shift-video` / `--shift-audio` / `--prompt` / `--prompt-file` / `--seed`
可单独覆盖。界面上跑用 `user/default/workflows/minimax_h3/t2va_8step_ui.json`
(侧边栏 Workflows → `minimax_h3`),选型依据和校验方式见那个目录的 README。

## 实测性能（1344×768，124 帧 ≈ 5.2s）

30 步基线：初始化 2m21s，采样 ~34s/step，**端到端 22m07s**。

慢的根因：33B dense transformer，且**开源版只有 full attention**（官方 sparse
attention 到 2026-08-14 仍未随权重发布）。所以提速只能靠砍步数 + 缩权重。

上了 `lightx2v/Minimax-h3-Turbo` 蒸馏 LoRA 之后（`LoraLoaderModelOnly` 直接打在
量化权重上，日志 `208 patches attached`，走在线重量化，**每步开销几乎为 0**）：

| 配置 | 冷启（起服后第一条） | 热跑 + 新 prompt | s/step |
|---|---|---|---|
| 30 步 int8（基线） | 22m07s | — | 34 |
| **8 步 int8 ← 定稿** | — | **292s / 310s / 330s** | ~33 |
| 8 步 w4a8 | — | 324s | ~36 |
| 4 步 int8（仅预览） | 531s / 502s | 179s / 176s / 180s | 32.5–33.9 |
| 4 步 w4a8（仅预览） | **394s** | 186s / 169s | 35.8–36.4 |

读法：

- **定稿是 8 步 int8，292–330s**。选它是因为画质≈30 步基线，见下面那张 lapvar 表；
  4 步虽然只要 176s 但丢浅景深，不作为出片配置。
- **冷启 w4a8 快 24%**（394s vs ~516s）—— staged 27.7GB（11956+15711MB）对
  45.9GB（19995+25882MB），少读一半权重。
- **热跑 4 步时两者打平在 ~177s**：w4a8 采样慢的那 12s 正好被它更快的 TE 编码抵掉。
- **w4a8 每步慢 8%**。sm_89 上原生的是 `int8_tensorwise`，4-bit 权重要解包，
  省的是内存不是算力。所以 8 步场景 w4a8 反而输 32s。
- 结论：**一次性 / 冷启用 w4a8，服务常驻批量出片用 int8**。同一 LoRA + 同 seed 下
  两种量化画质等价（背景 lapvar 31.0 vs 39.3）。

### 计时的坑

同 prompt + 同 seed 会命中 ComfyUI 输出缓存，`Prompt executed in 0.00`、10s 返回，
把对比全废掉。测性能必须换 `--prompt` 或 `--seed`。另外只改 shift 不改 prompt 时
文本编码节点仍然命中缓存 —— 那条路省掉的 TE 编码在真实场景是要付的。

### 4 步 LoRA 会丢浅景深 —— 所以定稿用 8 步

**这是选型的决定性依据**,比耗时重要。在背景那块常年虚焦的区域量 Laplacian 方差
(越低=虚化越彻底,同 prompt / 同 seed 42):

| 配置 | LoRA | steps | shift_v | 背景 lapvar |
|---|---|---|---|---|
| 30 步基线 | 无 | 30 | 12 | **35.0** |
| **8 步(定稿)** | 8step | 8 | 12 | **31.0** |
| 8 步 w4a8 | 8step | 8 | 12 | **39.3** |
| 4 步 | 4step | 4 | 6 | 107.1 |
| 4 步 + shift 12 | 4step | 4 | 12 | 78.0 |
| 4 步 LoRA 跑 8 步 | 4step | 8 | 12 | **116.8** |

- **8 步 LoRA ≈ 30 步基线**(31.0 vs 35.0):1327s 压到 292s,散景/皮肤纹理都对得上。
- **4 步 LoRA 把 prompt 里的 shallow depth of field 丢了**,背景近乎合焦,人脸是过度
  平滑的塑料感。
- **这是 LoRA 的属性,不是步数的属性。** 两种救法都失败:shift 6→12 只走 27% 的路
  (107→78,暗调回来了、散景圆盘没回来);给 4 步 LoRA 喂 8 步**反而最差**(116.8),
  它被绑死在训练时那个 4 步 schedule 上。**别再调 4 步 LoRA 的 shift 了。**
- 第二条 prompt(面包房特写)复现:背景带 lapvar 4 步 85.2 / 8 步 **16.0**。
- 同 seed 下**镜头本身也会变**:改 shift/steps 就改了 sigma schedule,轨迹整个走偏
  (基线那条人戴帽子、机位更远、霓虹字不同)。所以这几条不能当"同一镜头的画质对比"看,
  只能比风格倾向和细节水平。

### 4 步的音频不可靠

输出流本身都对（`1344x768 @24fps h264` + `32000Hz stereo aac`，两条都 5.167s），
但 4 步下音频分支没充分展开，安静/细微的音效会塌成近静音：

| 配置 / 内容 | mean | max |
|---|---|---|
| 30 步基线（萨克斯） | -17.9 | — |
| 4 步 int8（萨克斯） | -32.7 | -16.1 |
| 4 步 int8 + shift_audio=6（萨克斯） | -31.7 | -18.0 |
| 4 步 w4a8（萨克斯） | -17.0 | -3.8 |
| 4 步 w4a8（猫呼噜/钟摆/鸟鸣） | **-50.2** | **-33.2** |
| 4 步 w4a8（铁匠敲击） | -29.1 | -5.0 |
| **8 步 int8（萨克斯）** | **-19.8** | **-4.6** |
| **8 步 w4a8（萨克斯）** | **-20.9** | **-4.9** |

| 4 步 int8（面包房，安静内容） | -43.5 | -20.7 |
| 8 步 int8（面包房，安静内容） | **-48.5** | **-30.9** |

- `shift_audio=6` **没用**（-32.7→-31.7，噪声级差别），别浪费时间调它。
- 跨 prompt 的方差（-17 ~ -50）比任何配置差异都大，**不要**从单条 prompt
  推断量化或 LoRA 对音频的影响。
- ⚠️ **"8 步是音频的分界线"这个说法已作废。** 在萨克斯那条 prompt 内部它成立
  （4 步 LoRA 不管给 4 步还是 8 步、shift 6 还是 12，都卡在 -31~-33；8 步 LoRA 和
  基线在 -18~-21），但换到面包房那条 prompt 上 **8 步反而更小**（-48.5 vs -43.5），
  没复现。**音频响度目前没有可靠的配置杠杆** —— 安静内容就是会塌，只能改 prompt
  写法，或者像 ref2va 那样在图里插一个 `AudioAdjustVolume` 补增益。
- 定稿选 8 步的理由是**画面**（见上面那张 lapvar 表），不是音频。
- 8 步那个 LoRA 训练在 544p mixed，但 768p 出来画面反而更干净（散景更好），
  没有 off-distribution 的可见代价 —— 上面那张 lapvar 表证实了这点。

### 为什么没上 sage attention

当初按 4 步算，采样只占热跑的 ~75%、冷启的 ~27%，attention 优化最多吃采样的
20–30%（约 30s），所以搁置了。sensenova env 里 sageattention/flash_attn/xformers
都没装（只有 triton 3.7），要重编译。

⚠️ **定稿改成 8 步之后这个结论该重算**：8×33s = 264s 采样占 292s 热跑的 ~90%，
attention 优化的上限变成 50–80s（约 20%）。还没测，是目前最值得试的下一步。

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

5 秒 768P 一条 = $0.40。

**上 turbo LoRA 之后这笔账反过来了。** 之前 22 分钟一条，$2.24/h 的机器折算
$0.82/条，比 API 贵一倍；现在按定稿的 8 步算，292s 一条 = 12.3 条/h =
**$0.18/条**，比 API 便宜 2.2x。（4 步是 177s = $0.11/条、便宜 3.6x，但它丢浅景深，
只当预览用，不该拿来算出片成本。）

前提是**服务常驻、连续出片**。冷启一条要 394–530s，单条折算 $0.25–0.33，
零散跑就没这个优势了。自建另外的价值仍在可微调、无内容审核、无速率限制、离线可控。

视频资源包（$1000 起）目前不支持 H3，只能 pay-as-you-go。

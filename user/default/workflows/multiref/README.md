# 多参考图生图对比:Qwen-Image-Edit-2511 / FLUX.2-dev / SenseNova-U1

> 记录于 2026-07-28。用同一组参考图 + 同一条提示词横向对比三个**开源**模型的多参考图能力。

## 为什么是这三个

起因是评估阿里 `wan2.7-image`(多图参考、bbox 局部编辑、4K)。结论:**wan2.5 起全部转闭源 API**,
HF `Wan-AI` 组织的开源权重停在 Wan2.2 且只有视频模型,拿不到权重。所以改为在开源阵营里找对标:

| 模型 | 参数 | 许可 | 参考图上限 | 备注 |
| --- | --- | --- | --- | --- |
| Qwen-Image-Edit-2511 | 20B | Apache-2.0 | **3**(受节点槽位限制) | 2511 版重点改了人物一致性/图像漂移 |
| FLUX.2-dev | 32B | 权重非商用,**出图可商用** | 多张(链式,无硬上限) | 下载量最高 |
| SenseNova-U1-8B-MoT | 8B | 本地已有 | **6** | 另有 interleave 图文交错能力 |
| ~~wan2.7-image~~ | ? | **闭源 API** | 0–9 | ¥0.2/张,pro 版 ¥0.5/张支持 4K |

三者都不用装自定义节点 —— ComfyUI 原生支持(`nodes_qwen.py` / `nodes_flux.py` / `nodes_edit_model.py`)。

## 三者喂参考图的机制不同

这是对比时最需要注意的地方,不是简单的"参数不同":

- **Qwen**:参考图进 `TextEncodeQwenImageEditPlus` 的 `image1/2/3`。每张图会**同时**走两条路 ——
  缩到 384² 喂 Qwen2.5-VL 做视觉理解,再缩到 1024² 过 VAE 变成 `ref_latents`。
  节点只有 3 个槽位,所以上限是 3 张(见 `comfy_extras/nodes_qwen.py:63`)。
- **FLUX.2**:每张图各自 `VAEEncode`,然后用 `ReferenceLatent` **链式串联**
  (节点自述 "chain multiple to set multiple reference images",见 `comfy_extras/nodes_edit_model.py:12`)。
  张数不受槽位限制。注意 Flux2 latent 是 **128 通道**,必须用 `EmptyFlux2LatentImage`,
  且 sigmas 要用 `Flux2Scheduler`(按 seq_len 算,不是普通 scheduler)。
- **SenseNova**:`SenseNovaU1LocalCompose` 的 `image` + `image2..image6`,
  prompt 里用 `<image>` 占位符**按序绑定**每张图 —— 这点和另两个完全不同,
  不给占位符效果会明显变差。

## 用法

```bash
# 1. 准备权重(EBS 持久盘,正常只校验不重下)
bash user/default/scripts/restore_multiref_models.sh

# 2. 三个模型跑同一组图 + 同一 prompt + 同一 seed
python user/default/scripts/multiref_compare.py \
    --ref charA.png --ref charB.png \
    --prompt "the two people shaking hands in a modern office, photorealistic" \
    --sensenova-prompt "<image> 和 <image> 在现代办公室握手,写实摄影" \
    --outdir ./cmp

# 3. 只跑其中两个 / 调参
python user/default/scripts/multiref_compare.py -m qwen -m flux \
    --ref a.png --ref b.png --ref c.png --prompt "..." \
    --seed 123 --steps 30 --width 1024 --height 1024

# FLUX.2 挂 Turbo LoRA 少步数快出图(调 prompt 阶段省时间)
python user/default/scripts/multiref_compare.py -m flux --flux-turbo --steps 8 \
    --ref a.png --ref b.png --prompt "..."
```

脚本会把参考图**只上传一次**,三个模型共用同一批服务端文件,保证对比公平;
参数与耗时记录在输出目录的 `compare_meta.json`。单个模型失败不影响其余继续跑。

## 工作流 JSON

`*.api.json` 是 API 格式(可直接 POST `/prompt`,也可拖进 ComfyUI)。
它们由 `multiref_compare.py` 里同一套 builder 函数生成,所以不会与脚本漂移;
要重新生成就跑脚本里的 builder。**里面的图片名是占位符**(`ref_a.png` 等),
拖进 ComfyUI 后需要重新指定 LoadImage。

| 文件 | 模型 |
| --- | --- |
| `qwen_image_edit_2511_multiref.api.json` | Qwen-Image-Edit-2511,3 张参考图 |
| `flux2_dev_multiref.api.json` | FLUX.2-dev,3 张参考图链式 ReferenceLatent |
| `sensenova_u1_multiref.api.json` | SenseNova-U1 Compose,3 张参考图 |

## 显存(L40S 46G)

三个都是 fp8 量化版,单个模型能装下,但**不要同时加载**。
脚本是顺序跑的,ComfyUI 会自行换出上一个模型;若遇 OOM:

- Qwen/FLUX:降 `--width/--height` 到 768,或给 FLUX 加 `--flux-turbo` 减步数
- SenseNova:调小 `--input-mp`(每张输入图的像素上限,多图时最容易 OOM)

## 权重存放位置 ⚠️

**这两个模型在持久 EBS 盘 `/mnt/models`,不是临时盘。** 与既有模型的区别:

| 路径 | 类型 | stop→start |
| --- | --- | --- |
| `/opt/dlami/nvme` | instance store 临时盘 | **全部清空**(wan / SenseNova / hf_cache 都在这) |
| `/mnt/models` | EBS 300G `vol-0426c78922b2dcd49` | **保留** |

所以 stop→start 之后是**混合状态**:Qwen/FLUX 的软链还好使,
wan 和 SenseNova 的软链会断(`folder_paths.py:371` 会打 `doesn't link anywhere` 警告,
那些模型从下拉列表消失,但不会导致 ComfyUI 崩溃)。恢复各自跑:

```bash
bash user/default/scripts/restore_multiref_models.sh    # Qwen/FLUX(通常秒过)
bash user/default/scripts/restore_sensenova_models.sh   # SenseNova(要重下 33G/个)
bash user/default/scripts/download_wan22_weights.sh     # wan2.2
```

详见 `user/default/docs/NVME_STORAGE_NOTES.md`。

# Wan-Animate-2（动作迁移 / 角色替身）

2026-09-30 新建。把一段**驱动视频**的动作迁到一张**角色参考图**上。
替代 `wan2.2/` 里那套 Wan2.2-Animate（14B fp8 + KJ 节点），这次全部用核心节点 + 蒸馏基座。

```bash
bash user/default/scripts/start_comfy.sh
python /tmp/run_wf.py user/default/workflows/wan_animate_2/motion_transfer.api.json
```

`motion_transfer.api.json` 实测：81 帧 / 480×832 / **397s** → `output/video/wan_animate2_00001_.mp4`

## 模型

| 位置 | 文件 | 备注 |
|---|---|---|
| `diffusion_models/` | `wan_animate_2_distill_int8_convrot.safetensors` | 蒸馏版，10 步 LCM。非蒸馏版要另配 lightx2v LoRA |
| `text_encoders/` | `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | ⚠ **在临时盘 `/opt/dlami/nvme`，实例 stop 后会没** |
| `vae/` | `wan_2.1_vae.safetensors` | 等价于官方模板要的 `Wan2_1_VAE_bf16`，省 250M 不用重下 |
| `clip_vision/` | `clip_vision_h.safetensors` | 这次唯一新下的文件 |

重启后如果报找不到 umt5：`bash user/default/scripts/check_env_after_restart.sh` 看一下，
然后从 `Comfy-Org/Wan_2.1_ComfyUI_repackaged` 重新拉到 `/mnt/models/text_encoders/`（别再放临时盘了）。

## 采样配方（蒸馏版）

```
UNETLoader → WanAnimate2Cache(gpu, int8)          ← 官方模板就是 gpu/int8
           → ModelSamplingSD3(shift 5.0)
KSamplerSelect(lcm) + BasicScheduler(simple, 10, 1.0) + SamplerCustom(cfg 1.0)
           → TrimVideoLatent(trim_amount = WanAnimate2ToVideo 的第 4 个输出)
```

`TrimVideoLatent` 不能省：`WanAnimate2ToVideo` 会在前面塞若干条件帧，
第 4 个输出（`["w2v", 3]`）就是要裁掉的帧数，不裁的话开头会多一段参考图静帧。

## 三路 CLIPTextEncode 分别是什么

这是最容易接错的地方：

| 接到 | 写什么 |
|---|---|
| `positive` | **角色外观 + 背景**，官方两段式：`Character appearance description: ...` 换行 `Background description: ...` |
| `negative` | 标准 Wan 中文负向（色调艳丽、过曝、静态、多余的手指……） |
| `positive_pose` | **描述驱动视频本身**，比如「一段人物跳舞的动作参考视频」——不是要生成的画面 |

另外两路 CLIPVision：参考图整张过 `CLIPVisionEncode(crop='none')` 给 `clip_vision_output`；
驱动视频用 `ImageFromBatch(0, 1)` 取**第一帧**再编码，给 `clip_vision_output_pose`。

## 尺寸和时长

- 驱动视频先 `ResizeImageMaskNode` 缩到目标尺寸（默认 480×832 竖版），**输出尺寸由这里决定**；
- 角色参考图必须 resize 到**和驱动视频一样**的尺寸（用 `GetImageSize` 串过去，别手填）；
- ⚠ **帧率沿用驱动视频**。驱动视频是 60fps 时，81 帧只有 **1.35 秒**。
  想要 5 秒先把驱动视频抽成 16fps，或者把 `length` 调大（显存换时长）。

## 接长视频

`WanAnimate2ToVideo` 的 6 个输出依次是
`positive / negative / latent / trim_latent / trim_image / video_frame_offset`。

续拍：把上一段的 `video_frame_offset`（`["w2v", 5]`）回接到本段的 `video_frame_offset` 输入，
再把上一段解码出来的尾帧接到 `continue_motion`（optional 输入），动作就能无缝往后接。

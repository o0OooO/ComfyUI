# Video-Depth-Anything（视频 → 深度图视频）

2026-10-10 新建。输入一段视频，输出逐帧时序一致（不闪）的深度视频。
核心只有 Depth Anything 3（逐帧归一化，视频会闪），所以这里用第三方节点
[yuvraj108c/ComfyUI-Video-Depth-Anything](https://github.com/yuvraj108c/ComfyUI-Video-Depth-Anything)，
视频读写仍用核心的 `LoadVideo` / `CreateVideo` / `SaveVideo`。

```bash
bash user/default/scripts/start_comfy.sh
python /tmp/run_wf.py user/default/workflows/video_depth_anything/video_depth.api.json
```

换输入：改 `vid` 节点的 `file`（放在 `input/` 下）。

实测（L40S）：
- `test_ctrl.mp4` 17 帧 / 512×512 → 10s
- `dance_pose.webm` 379 帧 / 1080×1920 / 60fps → **90s**，显存峰值约 33G，输出 720×1280

## 模型

| 位置 | 文件 | 许可 |
|---|---|---|
| `models/videodepthanything/` → `/mnt/models/videodepthanything/`（EBS 持久盘） | `video_depth_anything_vitl.pth`（1.5G） | **CC-BY-NC-4.0，只能非商用** |

要商用换 `video_depth_anything_vits.pth`（Small，Apache-2.0）。
loader 下拉里列的文件不在本地时，节点会自己从 HF 下载到同一目录。

## 安装（换机器时）

节点是 git 子模块（钉在上游 `a0db08e`），`check_env_after_restart.sh` 第 2 / 2b 节会检查节点、依赖和权重。

```bash
git submodule update --init --recursive
~/miniconda3/envs/sensenova/bin/pip install -r custom_nodes/ComfyUI-Video-Depth-Anything/requirements.txt
ln -sfn /mnt/models/videodepthanything models/videodepthanything
```

依赖只新增 opencv / matplotlib / imageio / easydict / OpenEXR，不动 torch 和 numpy。

## 参数

- `max_res` 1280：长边超过就先缩小，**输出分辨率也是缩小后的**，不会放回原尺寸。
  要原尺寸就接一个 `ImageScale`，或者调大 `max_res`（显存跟着涨）。
- `input_size` 518：模型推理尺寸，一般不动。
- `colormap`：`gray` 近白远黑，给 ControlNet / 深度条件用；`inferno` 是给人看的伪彩。
- 帧率沿用输入视频，帧数和输入一致。
- 整段深度会以 float32 存在内存里，几千帧的长视频先抽帧或分段。

# Wan2.2 工作流(API 格式)

这里的 `*.api.json` 是 `scripts/wan22_api.py` 各 task **实际发给 ComfyUI 后端的工作流**
(API prompt 格式),由脚本 `--export-workflow` 导出,已验证可被 ComfyUI `/prompt` 接受。

## 9 个 task

| 文件 | 功能 | 关键输入(占位字段) |
|---|---|---|
| `t2v.api.json` | 文本生视频 | (仅 prompt) |
| `i2v.api.json` | 图生视频 | `INPUT_IMAGE.png` |
| `flf2v.api.json` | 首尾帧生视频 | `START_FRAME.png` / `END_FRAME.png` |
| `fun_inpaint.api.json` | 首尾帧补全 | `START_FRAME.png` / `END_FRAME.png` |
| `fun_control.api.json` | 控制视频驱动 | `CONTROL_VIDEO.mp4` (+可选 ref `INPUT_IMAGE.png`) |
| `vace.api.json` | 多参考 / 控制视频 | `REF_IMAGE.png` (+可选 `CONTROL_VIDEO.mp4`) |
| `phantom.api.json` | 多主体参考 | `REF_IMAGE.png` |
| `s2v.api.json` | 音频驱动 / 人物口型 | `INPUT_IMAGE.png` + `INPUT_AUDIO.wav` |
| `animate.api.json` | 动作驱动 | `INPUT_IMAGE.png` + `POSE_VIDEO.mp4` |

## 格式说明(重要)

- 这是 **API 格式**(`{节点id: {class_type, inputs}}`),**不能直接拖进 ComfyUI 网页界面**编辑;
  它用于程序化提交(`POST /prompt`)或作为连接关系的参考。
- 文件里的素材路径是**占位符**(如 `INPUT_IMAGE.png`、`REF_IMAGE.png`),
  实际使用时:先把素材上传到 ComfyUI 的 input 目录,再把对应 `LoadImage`/`LoadVideo`/
  `LoadAudio` 节点的文件名改成真实文件名。
- 多参考(vace/phantom)只导出了 1 个参考图节点;脚本运行时传多个 `--ref` 会用
  `ImageBatch` 链式合并出更多节点。

## 推荐用法

直接用脚本,不用手改 JSON(脚本会自动上传素材、按参数构建并提交):

```bash
# 见 scripts/wan22_api.py 头部注释
python ../../scripts/wan22_api.py i2v --image cat.png --prompt "..." -o out.mp4
python ../../scripts/wan22_api.py vace --ref a.png --ref b.png --prompt "..." -o out.mp4
```

重新导出某个 task 的工作流:
```bash
python ../../scripts/wan22_api.py <task> --prompt "..." --export-workflow <task>.api.json
```

## 默认参数

导出时用脚本默认值:640x640 / 81 帧 / 20 步 / cfg 3.5 / shift 8.0 / euler+simple。
权重文件名见各 json 的 `UNETLoader` 节点(fp8 优先;phantom/animate 用 Kijai 版,vace 用 Comfy-Org fp16)。
正式出片建议挂 lightx2v 加速 lora(`--lora-high --lora-low` + `--steps 4 --cfg 1.0`)。

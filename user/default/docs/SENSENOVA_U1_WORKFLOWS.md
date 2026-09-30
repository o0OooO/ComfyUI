# SenseNova U1 工作流说明（含 Imagen 2 功能平替）

> 记录于 2026-06-01。工作流文件位于 `user/default/workflows/sense-nova-u1/`。
> 所有自建工作流已在本机 ComfyUI 后端**端到端真实跑通验证**（提交 API、产出图片、人工核对合成方向）。

---

## 工作流清单

| 文件 | 类型 | 说明 |
|---|---|---|
| `t2i.json` | 官方 | 文生图（Text-to-Image） |
| `editing.json` / `SenseNova_U1_image_edit.json` | 官方 | 指令式整图编辑（mask-free 编辑） |
| `interleave.json` | 官方 | 图文交错生成 + 思维链 |
| `infographic.json` | 官方 | 信息图生成 |
| `api_u1_fast_t2i.json` | 官方 | 走 U1-Fast API 的文生图 |
| **`SenseNova_U1_inpaint.json`** | **自建** | **局部重绘（平替 Imagen 2 inpainting）** |
| **`SenseNova_U1_outpaint.json`** | **自建** | **扩图（平替 Imagen 2 outpainting）** |

---

## Imagen 2（img-2）功能平替对照

Google Imagen 2 的核心图像能力 → SenseNova U1 + ComfyUI 原生节点的覆盖方案：

| Imagen 2 能力 | 平替方案 | 工作流 |
|---|---|---|
| Text-to-Image | SenseNova U1 文生图节点 | `t2i.json` |
| Mask-free 编辑（整图指令编辑） | SenseNova U1 图片编辑节点 | `editing.json` |
| **Inpainting**（遮罩局部修改/移除/插入） | Edit + 遮罩羽化 + 合成回贴 | `SenseNova_U1_inpaint.json` |
| **Outpainting**（向外扩展画布） | Pad 画布 + Edit 续画 + 合成 | `SenseNova_U1_outpaint.json` |

---

## 关键设计原理（为什么要这样搭）

SenseNova U1 是 **指令式编辑模型**：它对整张图按文字指令重绘，**不像 Stable Diffusion 那样原生吃 mask + latent**。因此要实现 Imagen 2 那种"只改选区、其余像素不动"的精确局部编辑，思路是：

> 让 SenseNova 整图重绘 → 再用 ComfyUI 原生 `ImageCompositeMasked` 把结果**只贴回 mask 区域**，未选区用原图像素覆盖回来。

合成数学（已读源码确认）：`输出 = mask×编辑结果 + (1−mask)×原图`
- mask=1 → 用 SenseNova 编辑结果
- mask=0 → 保留原图原始像素（这是"保护未选区"的关键）
- `resize_source=True` → 自动把 SenseNova 输出缩放对齐到目标尺寸（SenseNova 输出尺寸可能与输入不同）

---

## 1. Inpaint 工作流（`SenseNova_U1_inpaint.json`）

```
SenseNovaU1LocalLoader ─┐
                        ├─► SenseNovaU1LocalImageEdit ─► (source)
LoadImage ──(IMAGE)─────┘                                  │
   │                                                       ▼
   ├──(IMAGE 原图)──────────────────────► ImageCompositeMasked ─► SaveImage
   │                                          ▲ (destination)
   └──(MASK 涂抹)─► GrowMask ─► FeatherMask ──┘ (mask)
```

**用法：**
1. `LoadImage` 载入图片，**右键 → Open in MaskEditor**，涂抹你想修改的区域。
2. 在编辑节点的 `prompt` 里写要在涂抹区生成什么（如 "a bouquet of red roses"）。
3. `GrowMask`（expand=6）+ `FeatherMask`（24px）让边缘过渡柔和、避免硬边。
4. 运行：只有涂抹区被替换，其余像素逐位保留原图。
5. 正式出图把编辑节点 `num_steps` 设回 **50**（验证时用的 2）。

> 节点里还接了一个 `PreviewImage` 显示 SenseNova 的原始整图重绘结果，方便和合成后效果对比。

---

## 2. Outpaint 工作流（`SenseNova_U1_outpaint.json`）

```
LoadImage ─► ImagePadForOutpaint ─┬─(padded IMAGE)─► Edit ─►(source)
                                  │                          │
SenseNovaU1LocalLoader ───────────┼──────────────────────────┤
                                  │                          ▼
                                  ├─(padded IMAGE dest)► ImageCompositeMasked ─► SaveImage
                                  └─(MASK: pad区=1)──────────┘ (mask)
```

**用法：**
1. `ImagePadForOutpaint` 设置四周扩展像素（默认各 256，feathering=40）。它输出扩展后的画布 + 一个 mask（**pad 区=1、原图区=0**，已含羽化）。
2. 编辑节点 `prompt` 写如何续画（如 "extend the scene naturally to fill the frame"）。
3. 合成：中心原图精确保留，仅扩展区用 SenseNova 生成内容填充。
4. 正式出图 `num_steps=50`。

> ⚠️ `ImagePadForOutpaint` 的 mask 极性是 **pad 区=1**，正好直接喂给 `ImageCompositeMasked` 当 mask，无需取反。

---

## 通用提示

- **num_steps**：验证/预览用 1-2 步，正式出图 50 步。L40S 上 50 步约 1 分钟级。
- **想看进度**：SenseNova 自己的采样循环不打印逐步进度条，看 `nvidia-smi` 利用率即可确认在跑（"Loading checkpoint shards 100%" 之后还有搬权重+采样，不是卡死）。
- **cfg_scale**：默认 4.0，越高越贴合指令；`img_cfg_scale` 越高越贴合原图（编辑场景默认 1.0）。
- **think_mode**：开启后模型先推理再生成，适合需要"理解因果/物理"的复杂编辑（更慢）。
- 这两个工作流只用了 ComfyUI **原生节点** + SenseNova 编辑节点，无需额外插件。

# LTX-2.3 多参考图生视频(Ingredients IC-LoRA)

传几张主角/道具/场景图,生成保持这些元素一致的短视频。

---

## 一、核心认知:不是"喂多张图"

一开始很容易以为像 Qwen-Image-Edit-2511 那样,多张参考图分别接进节点 —— **方向是错的**。

LTX-2.3 的多参考走的是 **reference sheet(参考图册)** 路线,靠
[`Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients`](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients)
这个 IC-LoRA(官方称 "reference sheet control"):

```
4 张参考图  →  拼成 1 张 2x2 图册(768x448)  →  loop 成 121 帧静态视频
            →  当 in-context 条件喂给模型  →  出片
```

模型从**参考 latent** 里读"元素长什么样",从**提示词**里读"发生什么事"。

工作流已经把拼图这步自动化了,你只管往 4 个 `Load Image` 里塞图。

---

## 二、不需要装第三方节点包

官方示例工作流
(`LTX-2.3_ICLoRA_Ingredients_Single_Stage_Distilled.json`)依赖
`ComfyUI-LTXVideo` 节点包里的 `LTXAddVideoICLoRAGuide` / `LTXICLoRALoaderModelOnly`。

读过那个包的源码后确认:它本质是包了一层核心的 `LTXVAddGuide.append_keyframe`,
而这个 LoRA 的 `reference_downscale_factor = 1`(官方模型卡写明),
所以**专用加载器没有额外作用**。

本工作流用 `LTXVAddGuide` + `LoraLoaderModelOnly` 等价实现,**纯核心节点,零额外依赖**。

---

## 三、准备

### 1. 权重

```bash
bash user/default/scripts/restore_ltx23_models.sh
```

| 文件 | 大小 | 落盘位置 |
|---|---|---|
| `ltx-2.3-22b-dev-fp8.safetensors` | 28G | 临时盘 ⚠ |
| `gemma_3_12B_it_fp4_mixed.safetensors` | 8.8G | 临时盘 ⚠ |
| `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | 950M | 临时盘 ⚠ |
| `ltx_2.3_22b_distilled_1.1_lora_...bf16.safetensors` | 2.6G | 临时盘 ⚠ |
| `ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors` | 1.3G | **持久盘** /mnt/models |

> ⚠ 前 4 个在 `/opt/dlami/nvme`,**stop→start 会被整盘清空**,重启后必须重跑上面的脚本
> (约 40G 重新下载)。新增的 IC-LoRA 特意放了持久 EBS,不用重下。

### 2. IC-LoRA 是 gated 仓库

必须两步都做到:

1. 浏览器打开
   https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients
   点 **Agree and Access**(一次性)
2. 本机放 token:`hf auth login`,或 `export HF_TOKEN=hf_xxx`

没 token 时恢复脚本会跳过它并提示,其余 4 个照常恢复。

### 3. 显存

`custom_nodes/ComfyUI-SenseNova-U1` 会把模型缓存在插件私有的 `_LOCAL_MODEL_CACHE`
全局 dict 里,**ComfyUI 的 `/free` 接口管不到**(实测调了返回 200 但显存不降)。

跑过 SenseNova 之后想跑 LTX,得**重启 ComfyUI** 才能释放:

```bash
bash user/default/scripts/start_comfy.sh restart
nvidia-smi --query-gpu=memory.used --format=csv   # 应降到 ~400MB
```

22B fp8 + 双 LoRA + 参考 token 翻倍,768x448x121 在 46G L40S 上跑得动,但要先腾空显存。

---

## 四、用法

打开 `user/default/workflows/video_ltx2_3_multiref_ingredients.json`。

### 4 个参考位

| 位置 | 放什么 |
|---|---|
| 左上 | 主角 A —— 正脸特写 + 转身图最佳 |
| 右上 | 主角 B(或主角 A 的第二视角) |
| 左下 | 关键道具 —— 产品图风格,多角度 |
| 右下 | 场景 —— 干净的环境全景 |

- 参考图**黑底、无文字**,主体占满画面。
- 只有 2-3 张参考:把用不到的那路 `Image Stitch` 用 **Ctrl+B 旁路**掉。
- 先跑一次看 `图册预览` 节点,确认拼图正确、主体没被裁掉,再去调提示词。

### 提示词:必须两段式

这是**最影响效果**的一点,结构要和训练时一致:

```
### Reference Sheet Description
**Top Row Left (Character):** <逐格描述外观:发型、瞳色、肤色、整套服装的颜色材质>
**Top Row Right (Character):** <同上>
**Bottom Row Left (Prop):** <道具,产品图口吻>
**Bottom Row Right (Setting):** <场景:建筑、光线、背景层次>

### Target Description
<镜头景别与运动 → 谁在场穿什么 → 逐拍动作 → 台词(带引号) → 光线与环境音>
```

关键:**Target 段里要用图册里同样的措辞**把角色和道具再描述一遍,模型才会把它们绑定到参考上。
图册决定"长什么样",Target 决定"发生什么"。

官方建议负向提示词:`worst quality, inconsistent motion, blurry, jittery, distorted`

---

## 五、分辨率别乱改

**768x448 / 121 帧 / 24fps 是官方唯一训练 bucket**,模型卡明确写了其他分辨率是
out of distribution。约束:宽高能被 64 整除,帧数为 8n+1,参考视频帧数 >= 121。

真要改,得同步改 4 处:
`Empty LTXV Latent Video`、`Repeat Image Batch` 的 amount、
`LTXV Empty Latent Audio` 的 frames_number、4 个 `Resize and Pad Image`(各为总尺寸一半)。

推荐直接改生成器脚本顶部的 `W / H / FRAMES / FPS` 后重跑,它带节点和连线校验:

```bash
python3 user/default/scripts/gen_ltx23_multiref_workflow.py
```

---

## 六、调优

| 症状 | 处理 |
|---|---|
| 角色脸/衣服漂移 | 该格给正脸特写+转身图,别塞太满;IC-LoRA 强度提到 1.2–1.4 |
| 某道具没出现 | 图册必须有独立一格,且在 `Reference Sheet` 段落里写出来 |
| 参考跟太死、不动 | `LTXVAddGuide` 的 strength 降到 0.8–0.9 |
| 想更高保真 | 重要元素的格子给大些 —— **占的面积越大,还原越好** |

### 快档 vs 高质量档

当前默认是 **8 步蒸馏档**(快):`ManualSigmas` 8 步 + `CFGGuider` cfg=1 + 蒸馏 LoRA 0.5。

切官方推荐的**高质量档**(30 步):
1. 蒸馏 LoRA 节点 Ctrl+B 旁路
2. `CFGGuider` 的 cfg 改 **4.0**
3. `ManualSigmas` 换成 30 步:
   ```
   1.0, 0.966, 0.933, 0.9, 0.866, 0.833, 0.8, 0.766, 0.733, 0.7, 0.666, 0.633, 0.6, 0.566, 0.533, 0.5, 0.466, 0.433, 0.4, 0.366, 0.333, 0.3, 0.266, 0.233, 0.2, 0.166, 0.133, 0.1, 0.066, 0.033, 0.0
   ```

官方模型卡另提到验证时用了 STG(spatiotemporal guidance,mode `stg_v`,block 29,scale 1.0)
有助运动稳定 —— 但 STG 需要 `ComfyUI-LTXVideo` 节点包,本工作流没用。

---

## 七、相关文件

| 文件 | 作用 |
|---|---|
| `user/default/workflows/video_ltx2_3_multiref_ingredients.json` | 工作流 |
| `user/default/scripts/gen_ltx23_multiref_workflow.py` | 生成器(带校验),改配置后重跑 |
| `user/default/scripts/restore_ltx23_models.sh` | 权重恢复/校验 |
| `user/default/scripts/start_comfy.sh` | 启停 ComfyUI(释放显存) |
| `user/default/docs/NVME_STORAGE_NOTES.md` | 三块盘的布局与坑 |

## 八、验证状态

**已端到端实跑通过** ✓

| 项目 | 结果 |
|---|---|
| 图册拼装 | 真实图实跑,输出正好 768x448、黑边填充正确 |
| 图结构 | 38 节点 / 45 连接,节点名/输入名/类型/槽位全项校验通过 |
| 提交时 node_errors | 0 |
| 生成耗时 | **162 秒**(8 步蒸馏档) |
| 峰值显存 | 约 37.5G / 46G(L40S 跑得下) |
| 输出规格 | 768x448 / 121 帧 / 24fps / 5.04s / h264 + AAC 音轨 |
| 多参考是否生效 | **是** —— 角色身份跨 121 帧稳定,场景明显取自图册的场景格 |
| IC-LoRA metadata | 从权重确认 `reference_downscale_factor = 1`,证实无需专用加载器 |

### 实测踩到的一个坑(印证官方说法)

冒烟测试时提示词写的是 cream wool coat(米色羊毛外套),生成出来是**粉色**。
原因:那次用的是随便挑的 4 张图,**图册里根本没有米色外套那一格**,
且相关那格主体占比很小。

正好对上官方两条 tips:
- 模型只还原**图册里存在**的元素 —— 要什么就得给它一格
- 元素占的**面积越大**,还原越忠实

所以提示词里写了但图册里没有的东西,模型会自由发挥。

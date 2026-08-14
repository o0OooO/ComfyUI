# MiniMax-H3 工作流(t2va / ref2va)

两条流程,两套 transformer,采样参数**不通用**:

| | t2va(纯文本生视频+音频) | ref2va(多参考) |
| --- | --- | --- |
| 节点 | `MiniMaxH3ImageToVideo` | `MiniMaxH3ReferenceToVideo` |
| 权重 | `minimax_h3_fl2va_pruned_int8_convrot` | `minimax_h3_hybrid_fl2va_ref2va_b25-49` |
| 界面用 | `t2va_8step_ui.json` | `ref2va_hybrid_ui.json` |
| 脚本 | `gen_minimax_h3_t2va.py` | `gen_minimax_h3_ref2va.py` |

视频和 32kHz 立体声在同一次前向里联合生成(含对白口型),不是后期配音。
基础部署见 [../../docs/MINIMAX_H3_GUIDE.md](../../docs/MINIMAX_H3_GUIDE.md)。

---

# t2va:用 8 步蒸馏 LoRA(2026-08-14 定稿)

> 六个配置横向对比(同 prompt / 同 seed 42 / 同 1344×768×124 / euler+simple,
> 只差 steps、shift、LoRA、量化)。结论:**留 8 步 LoRA + int8**,
> 端到端 292s,画质≈30 步基线的 1327s。

## 为什么是 8 步而不是 4 步

4 步那个 LoRA **会把 prompt 里的浅景深丢掉**。在背景那块常年虚焦的城市光带上量
Laplacian 方差(越低=虚化越彻底):

| 配置 | LoRA | steps | shift_v | 背景 lapvar |
| --- | --- | --- | --- | --- |
| 30 步基线 | 无 | 30 | 12 | **35.0** |
| **8 步(定稿)** | 8step | 8 | 12 | **31.0** |
| 8 步 w4a8 | 8step | 8 | 12 | **39.3** |
| 4 步 | 4step | 4 | 6 | 107.1 |
| 4 步 + shift 12 | 4step | 4 | 12 | 78.0 |
| 4 步 LoRA 跑 8 步 | 4step | 8 | 12 | **116.8** |

**这是 LoRA 的属性,不是步数的属性** —— 两种救法都失败:把 shift 从 6 提到 12 只走了
27% 的路(107→78,暗调回来了、散景圆盘没回来);给 4 步 LoRA 喂 8 步**反而更差**
(116.8,全场最烂),它被绑死在训练时那个 4 步 schedule 上。

第二条 prompt(面包房特写,prompt 明写 "very shallow depth of field")复现了同一结论:
背景带 lapvar 4 步 85.2 / 8 步 **16.0**,4 步那条能看清后厨每个人的帽子。
画面上还有个副作用:4 步的皮肤是过度平滑的塑料感,8 步和基线是正常纹理。

**8 步 LoRA 基本就是基线的等价物**(31.0 vs 35.0)。所以 4 步只留着看构图/试 prompt。

音频不在这个结论里:同一条 prompt 内 4 步稳定在 −31~−33 dBFS、8 步 −19.8,但换到
面包房那条 prompt 8 步反而更小(−48.5 vs −43.5)。**跨 prompt 方差压过配置差异,
别拿音量当选型依据** —— 详见 guide 里那张表。

## 用法

**界面上**:侧边栏 Workflows → `minimax_h3` → **`t2va_8step_ui`** → Run。
改 prompt 直接改 `MiniMaxH3ImageToVideo` 上那个多行框(t2va 这份没有连线进来的 widget,
和 ref2va 那份不一样,可以直接改)。

**命令行**:

```bash
python user/default/scripts/gen_minimax_h3_t2va.py                    # 定稿:8 步 + int8
python user/default/scripts/gen_minimax_h3_t2va.py --preset 4step     # 预览用,景深会塌
python user/default/scripts/gen_minimax_h3_t2va.py --preset base      # 30 步基线
python user/default/scripts/gen_minimax_h3_t2va.py --quant w4a8       # 冷启一次性跑
python user/default/scripts/gen_minimax_h3_t2va.py --prompt "..." --seed 7
python user/default/scripts/gen_minimax_h3_t2va.py --write-only       # 只重生成两个 JSON
```

`--steps` / `--shift-video` / `--shift-audio` 可单独覆盖,用来做 A/B;
一旦覆盖了就**不会**回写 git 里那两个 JSON(避免把定稿冲掉)。

## 两个 JSON 都是脚本生成的,别手改

`--write-only` 会同时写 `t2va_8step.api.json` 和 `t2va_8step_ui.json`,并做两道校验:

1. 所有节点名/输入名/combo 取值对着实时 `/object_info` 合法(和 ref2va 脚本同一个 `validate()`)
2. UI 版每个**有效值**逐项等于 api 版(`check_ui_matches_api()`),连 widget 顺序都是从
   `/object_info` 的声明顺序推出来的,不会和装着的 ComfyUI 漂移

跑过真机验证:直接 POST `t2va_8step.api.json` 出来的帧和当初挑中的那条
**逐字节相同**,音频也相同(mean −19.8 / peak −4.6 dBFS)。

---

# ref2va:多参考生视频

> 记录于 2026-08-13。同一组参考图 + 同一条 prompt + 同一 seed(42),横向对比官方
> ref2va checkpoint 和社区 hybrid 合并版,择优保留一个。结论:**留 hybrid**。

跑的是 `MiniMaxH3ReferenceToVideo` 节点:≤9 张图 / ≤3 段视频 / ≤3 段音频,总计 ≤12 个文件。

## 第一个坑:ref2va 是**另一套** transformer

本机原有的 `minimax_h3_fl2va_pruned_int8_convrot.safetensors` **跑不了多参考**。
HF 上游 `MiniMaxAI/MiniMax-H3` 里是 `transformer/`(fl2va)和 `transformer_ref/`(ref2va)
两个并列目录,`config.json` 完全相同、权重不同。想做多参考必须另下一份 19.5GB。

## 两个候选,以及为什么敢直接换

社区反馈官方 ref2va 有训练质量问题,裸画质不如 fl2va。
`smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models` 的思路是:两份权重除了每个 block 的
`adaln_proj`(模态/参考路由)之外余弦相似度 ≥0.9997,所以拿 fl2va 做底、只把后段
block 的 `adaln_proj` 换成 ref2va 的。

下载前后都做了校验,不是照抄说明:

- **量化布局一致**(下载前用 safetensors header 的 byte-range 读):三份文件都是
  932 个 tensor、`F32 210 / BF16 220 / F16 102 / I8 200 / U8 200`,同为 int8_convrot,可直接替换。
- **合并方式与命名相符**(下载后逐 tensor 哈希):`blocks.0/5/24.adaln_proj` 字节等于 fl2va,
  `blocks.25/30/48/49.adaln_proj` 字节等于 ref2va,非 adaln 权重(如 `blocks.30.attn.out_proj`)
  等于 fl2va。`b25-49` 这个名字是准确的,既不是单纯的 fl2va 也不是单纯的 ref2va。

## 对比结果(seed 42,同 prompt,同两张参考图)

联系表 `cmp_h3_ref2va/compare_seed42.png`(仓库根目录,未入 git),两条 mp4 也在那儿。

**hybrid 赢在三点,这三点决定了取舍:**

1. **人物 identity** —— hybrid 全程 5 秒脸都在画面里且稳定贴合参考图(脸型、发色、
   卷边袖 + 棕皮带 + 沙色长裙)。official 前 2.5 秒把她拍成背影(identity 无从判断),
   转正后脸偏圆、发色偏草莓金,离参考图更远。
2. **光** —— 参考图是金色时刻逆光暖调,hybrid 复刻了;official 天空偏灰、整体偏平。
   调色板距离(Bhattacharyya,对参考图取最近)0.348 vs 0.408 也支持这个观感。
3. **prompt 依从** —— prompt 写的是"驾驶门敞开、她把右前臂搭在车窗框上",hybrid 照做;
   official 车门始终关着,她靠在车身上。

**official 赢在两点,但都不致命:**

- **车辆 identity 更好**:official 的红色 C10 全程完整入画,深红车漆、锈斑、镀铬格栅、
  双圆灯、车斗徽标都贴参考图。hybrid 镜头一直咬着人,车多半只露门和后翼子板;
  1.0–1.5s 那块橙黄是逆光过曝的后翼子板(2.5s 之后车门恢复正常深红),不是 identity 崩了。
  → 但这说明 hybrid 换掉 `adaln_proj` 后,**非人物参考的绑定可能变弱**,做产品/车/道具
  多参考时值得再验一次。
- **音频响 8dB**:official mean −23.8 / peak −4.3 dBFS,hybrid mean −32.2 / peak −11.5。
  两条都有真实对白(语音带 300–3400Hz 能量突起,动态范围 25.1 vs 22.0 dB),hybrid 只是小声。
  **这个是可修的**,已在工作流里修掉(见下)。

没被采信的指标:高频细节能量(Laplacian 方差)official 47.5 / hybrid 28.2,看着 official 更"锐",
但 official 那一版锈迹纹理占画面更多、天空更平,而 hybrid 是 prompt 要求的浅景深 + 金色雾感,
这个数字在这里分不出好坏,所以没拿它当依据。时序抖动 hybrid 略稳(0.0575 vs 0.0618)。

## 音量:图里加了一个 +8dB

hybrid 的联合音频偏小,所以 `decode_audio` 和 `CreateVideo` 之间插了
`AudioAdjustVolume`(dB 整数),默认 `+8`:正好补到 official 的电平,且 peak
−11.5 + 8 = −3.5 dB **不会削顶**。单独跑节点验证过:−32.7/−12.3 → −24.7/−4.3 dB。
不想要就 `--audio-gain 0`,那个节点会整个不生成。

## 采样参数和 t2va 不一样,别照抄

| | t2va | **ref2va** |
| --- | --- | --- |
| sampler | `euler` | **`res_multistep`** |
| scheduler | `simple` | **`beta`** |
| steps | 8(8 步 LoRA) | **20**(未上 turbo) |
| SigmaShift | 12.0 / 3.0 | 12.0 / 3.0(不变) |

官方 R2V 模板自带的注释说:参考图很多的 prompt 上 `beta`/`normal` 明显好过 `simple`
(注释这么写,但它自己的 widget 出厂是 `simple`,得手动改)。

`MiniMaxH3SigmaShift` 在 12.0/3.0 上其实是 **no-op**,加不加一样:
`supported_models.py` 里 `MiniMaxH3.sampling_settings = {"shift": 12.0}` 已经给了采样器
同样的 shift,DiT 的 `sigma_shift_video=12.0 / sigma_shift_audio=3.0` 也是同样的默认值
(`comfy/ldm/minimax/model.py:418`,transformer_options 里没有才回落到它)。
所以官方 UI 模板完全不放这个节点是对的;`*.api.json` 里放着只是想显式写死,想调才有意义。

## prompt 必须写成官方六段式

这是**最大的质量杠杆**,比换 checkpoint 影响大。段落顺序固定,来自模型卡的
`docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md`:

```
subject_definitions:      # <Subject N> 是什么,以及它来自哪张素材
summary:                  # [task] + 一句话目标
retention_analysis:       # 每个 Subject 保留什么、重新生成什么
detailed_description:     # [Shot N] 逐镜头写画面 + 对白 + 动作
overall_soundscape:       # 环境音、动效
non_diegetic_music:       # 配乐(画外)
```

关键约定:

- **`<Subject N>` 和 `<Picture N>` 要分开**。`<Subject N>` 是可复用的实体,
  `<Picture N>`/`<Video N>`/`<Audio N>` 只是承载它的素材。分开写才能"借人物但不继承构图" ——
  写死成 `<Picture 1>` 容易把原图的机位姿势一起抄过来。序号按类型 1-based。
- 保留度标记:视觉 `fully_preserved` / `partially_preserved` / `attribute_transfer` /
  `weak_reference`;音频 `fully_copy` / `partially_copy` / `reference` / `weak_reference`。
- 任务前缀用 ` + ` 连接:`reference generation`、`video editing`、`audio reuse`、
  `audio reference`、`keyframe completion`、`video continuation`。
- 对白 `<d>[English] ...</d>`,说话人 `(S1)` / `(S2)`,镜头 `[Shot N] At MM:SS.mmm`。
- 画面和声音**写在同一段里**,模型是联合建模的。

脚本里 `PROMPT` 常量就是一个完整可抄的例子(reference generation,两个人/物 + 一句对白)。

## 参考图先处理再喂

- **水印必须裁掉**。原图是可灵出的,底部有"可灵AI 3.0 Omni"角标,直接当参考图会被
  当成参考信号学进输出。`input/h3ref_*.png` 是裁掉底部 140px 之后再 LANCZOS 缩到
  1408×736(/32 对齐,≈1.03MP)的。
- `ref_image_size="match"` 把每张参考图按面积等比缩到生成画布(只缩不放);`"max"` 用
  2048 短边,identity 更好但慢数倍 —— 参考 token 每个采样步都参与全注意力。
- 参考 token 其实不贵:两张 1408×736 各占 1012 行((1408/16/2)×(736/16/2)),合计 2024 行;
  目标视频是 37×1008 = 37296 行(`video_latent_t(124)=37`)。也就是 **+5% 序列长度**。

## 用法

**界面上**:ComfyUI 侧边栏 Workflows → `minimax_h3` → **`ref2va_hybrid_ui`** → Run。
参考图换成自己的就改两个 LoadImage;prompt 改左下那个 `PrimitiveStringMultiline`。

**命令行**:

```bash
# 权重(hybrid 已在 /mnt/models,正常不用重下)
ls -lL models/diffusion_models/minimax_h3_hybrid_fl2va_ref2va_b25-49.safetensors

# 默认:hybrid + 内置六段式 prompt + input/h3ref_{woman,truck}.png
python user/default/scripts/gen_minimax_h3_ref2va.py

# 自己的参考图 + 自己的 prompt(≤9 张)
python user/default/scripts/gen_minimax_h3_ref2va.py \
    --ref myA.png --ref myB.png --ref myC.png \
    --prompt-file /path/to/six_section_prompt.txt --seed 123

# identity 优先(慢数倍)/ 不加音量 / 只写 JSON 不提交
python user/default/scripts/gen_minimax_h3_ref2va.py --ref-image-size max
python user/default/scripts/gen_minimax_h3_ref2va.py --audio-gain 0
python user/default/scripts/gen_minimax_h3_ref2va.py --write-only
```

脚本会拿实时 `/object_info` 校验节点名、输入名和 combo 取值,所以 JSON 不会和装着的
ComfyUI 静默漂移。参考图放 `input/`。

⚠️ `input/` 和 `cmp_h3_ref2va/` 都**不在 git 里**(`/input/` 被 .gitignore 挡掉,
对比产物是二进制没必要入库)。换机器后 `input/h3ref_*.png` 需要重做:
取两张同一人物/车的图,裁掉底部水印,缩到 1408×736。

## 文件

都在 `user/default/workflows/minimax_h3/`,ComfyUI 侧边栏 **Workflows** 里会显示成
一个 `minimax_h3/` 分组,点开即用(不用拖文件)。

| 文件 | 格式 | 说明 |
| --- | --- | --- |
| `t2va_8step_ui.json` | **UI** | **t2va 界面上用这个**。8 步 LoRA / int8 / shift 12-3 / euler+simple / 1344×768×124 都已填好 |
| `t2va_8step.api.json` | API | t2va 脚本/自动化用。就是实测挑中的那份(逐字节复现过) |
| `ref2va_hybrid_ui.json` | **UI** | **ref2va 界面上用这个**。开箱即用:hybrid 权重 / beta / 1344×768 / +8dB / 六段式 prompt 都已填好,点 Run 就跑 |
| `ref2va_hybrid.api.json` | API | 脚本/自动化用,直接 POST `/prompt`。就是实测跑出结果的那份 |
| `ref2va_official.api.json` | API | 对比基线,留作复现记录。官方权重已删,要跑得先重下(见下) |
| `video_minimax_h3_r2v_official.json` | UI | 官方 Comfy-Org R2V 模板原版,**没动过**,留作对照。**装着的 `comfyui_workflow_templates` 包里没有这个模板**,是从 GitHub raw 抓的 |

`ref2va_hybrid_ui.json` 是从官方模板改的,改了 5 处(模型 / scheduler / 分辨率 /
+8dB / prompt),清单写在图里那个「本机改动」note 上。校验过:link 表完整、widget 取值
对着实时 `/object_info` 合法,并且**逐个有效值和 `ref2va_hybrid.api.json` 完全一致** ——
包括穿过 `ResolutionSelector`(0.98MP→1344×768)和 `ComfyMathExpression`(5s→124 帧)算出来的值。
换句话说界面这份跑出来的就是实测那条。

⚠️ UI 版里 width/height/length/prompt 都是**连线进来的**,直接改
`MiniMaxH3ReferenceToVideo` 上的 widget **没用**(连线时 widget 值被忽略)。要改:

- prompt → 改左下 `PrimitiveStringMultiline`
- 分辨率 → 改 `Resolution Selector` 的 megapixels(**0.98** = 1344×768;1.0 会变 1376×768 超上限)
- 时长 → 改 `PrimitiveFloat`(秒),后面的 `ComfyMathExpression` 自动对齐到 17k+5 网格

`*.api.json` 由 `gen_minimax_h3_ref2va.py` 里的 builder 生成,改参数就重跑脚本,别手改 JSON。
里面的图片名是 `h3ref_*.png`,拖进 ComfyUI 后按需重新指定 LoadImage。

## 权重

只留了 hybrid。official 已删(省 20GB),要复现基线的话 ~4 分钟能下回来:

```bash
HF_HOME=/mnt/models/hf_cache hf download Comfy-Org/MiniMax-H3 \
    diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors \
    --local-dir /mnt/models/diffusion_models/.hf_h3
mv -n /mnt/models/diffusion_models/.hf_h3/diffusion_models/*.safetensors \
      /mnt/models/diffusion_models/
ln -s /mnt/models/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors \
      models/diffusion_models/
python user/default/scripts/gen_minimax_h3_ref2va.py --model official --audio-gain 0
```

`HF_HOME` 一定要指到 `/mnt/models/hf_cache`:根盘只剩 ~22GB(93%),默认缓存位置会把它填满。
权重放 EBS 持久盘 `/mnt/models`,`models/` 下逐文件软链 —— 见 [[ebs-persistent-model-disk]] 约定。

## 实测耗时(L40S 48GB,1344×768,124 帧≈5.17s,20 步)

| | 初始化 | 采样 | 端到端 |
| --- | --- | --- | --- |
| official | ~6.4 min | 13m06s(39.3 s/step) | 19m30s |
| hybrid | ~2.9 min | 13m07s(39.4 s/step) | 16m00s |
| hybrid(+8dB,最终版) | ~5.7 min | 13m05s(39.0 s/step) | 19m45s |

**采样恒定 13 分钟(39 s/step),波动全在初始化**,是磁盘冷热不是模型差异:系统内存 61GB
装不下 TE 27GB + DiT 20GB 两份 page cache,轮换着跑就得回 EBS 重读 19.5GB。所以别拿端到端
时间比模型。

固定 seed 是确定性的:hybrid 两次跑出来的视频流 md5 完全一致(`658c60f1…`),
+8dB 只动音频轨。

## 第二个坑:必须先重启 ComfyUI 再跑

第一次提交直接 OOM,栈在 `int8_linear` 的 `torch.cat`(MLP fc1),第 0 步就炸。

原因不是参考 token 太贵(只 +5%),而是**那个 server 已经连续跑了 8 天**,期间加载过
SenseNova-U1 和 ACEStep,OOM 后还有 34GB 显存没释放。SenseNova 节点把模型缓存在插件私有的
`_LOCAL_MODEL_CACHE` 里,ComfyUI 的 `/free` 管不到 —— 只能重启进程。重启后干净起来是 435MiB,
两条都一次跑过,峰值 35GB。

```bash
# 注意:start_comfy.sh 的 pgrep 匹配串是 "main.py --listen .* --port 8188",
# 你自己的命令行里带上这串会被它连带杀掉(踩过)。
bash user/default/scripts/start_comfy.sh restart
```

跑 H3 之前 `nvidia-smi` 确认显存是干净的,不然 20 分钟白等。

## 约束(和 t2va 相同)

- `length` 必须落在 17k+5 网格:124 / 141 / 158 …,124 帧 = 5.17s,训练范围约 124–362 帧。
- 画布 768 短边,面积上限 768×1344,每轴对齐 32。1344×768 是原生尺寸。
- **H3-Context-IR 没开源**,官方说它对最终质量至关重要。自建只有 H3-Base,
  所以 prompt 得自己按六段式写(或付费调 API 让它改写)。
- `lightx2v` 的 Prompt-Rewriter-LoRA **只支持纯文本,不支持参考图**,这里用不上。
- 许可证是 `minimax-h3-community-license-agreement`,不是 Apache。

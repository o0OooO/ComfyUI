# 模型权重存储与恢复说明（临时盘方案）

> 记录于 2026-06-02。本机是 EC2 DLAMI 实例,自带一块 419G 本地 NVMe 临时盘。

## 背景

根盘(EBS 300G)空间被模型占满。为腾空间,所有大模型权重已挪到本地 NVMe
临时盘 `/opt/dlami/nvme`,用软链接回 ComfyUI 的 models 目录。

## ⚠️ 最重要的一点:临时盘会丢数据

`/opt/dlami/nvme` 是 EC2 **instance store(本地实例存储)**:
- **reboot(重启)** → 数据保留 ✓
- **stop / start(停止后再启动)** → **数据全部清空** ✗
- **terminate(终止)** → 数据全部清空 ✗

所以每次 **stop→start 后,所有挪过去/下载的模型都没了,软链变成断链**,需要重新恢复。

## 临时盘布局

```
/opt/dlami/nvme/
├── comfy_models/models/...   # 从根盘挪来的现有模型(LTX/z_image/gemma/Wan i2v/loras)
│   ├── checkpoints/          #   + ace_step_1.5_turbo_aio.safetensors(9.4G,音乐,ComfyUI 原生)
│   │                         #   + stable_audio_3_small_sfx.safetensors(2.2G,音效,ComfyUI 原生)
│   └── text_encoders/        #   + t5gemma_b_b_ul2.safetensors(1.2G,Stable Audio 3 SFX 所需)
├── hf_cache/huggingface/     # 共享 HF 缓存,软链自 ~/.cache/huggingface
│                             #   - SenseNova-U1-8B-MoT(基座,~33G,通用 t2i/编辑)
│                             #   - SenseNova-U1-8B-MoT-Infographic(信息图/海报专精,~33G)
│                             #   ⚠️ 两个都是 8B,显存(L40S 46G)一次只能加载一个,切换用
└── wan_weights/              # 新下载的 Wan2.2 各 task 权重
    ├── diffusion_models/
    └── audio_encoders/
```

软链关系:
- `ComfyUI/models/<sub>/<file>` → `/opt/dlami/nvme/comfy_models/models/<sub>/<file>`
- `ComfyUI/models/diffusion_models/<wan file>` → `/opt/dlami/nvme/wan_weights/diffusion_models/<file>`
- `ComfyUI/models/audio_encoders/<file>` → `/opt/dlami/nvme/wan_weights/audio_encoders/<file>`
- `~/.cache/huggingface` → `/opt/dlami/nvme/hf_cache/huggingface`

## stop→start 之后怎么恢复

实例重新启动后,临时盘是空的(甚至挂载点可能要重建)。按需重新下载:

```bash
cd /home/ubuntu/projects/mead/ComfyUI

# 1. 确认临时盘已挂载(DLAMI 通常自动挂)
df -h /opt/dlami/nvme || echo "临时盘未挂载,需手动挂载"

# 2. 重新下载 Wan2.2 各 task 权重(会重建软链)
bash user/default/scripts/download_wan22_weights.sh          # 全部
# 或只下需要的: bash user/default/scripts/download_wan22_weights.sh t2v s2v

# 3. SenseNova 权重:一键恢复(重建软链 + 重新下载基座和 Infographic,各约 33G)
bash user/default/scripts/restore_sensenova_models.sh        # 两个都恢复
# 或只恢复需要的: bash user/default/scripts/restore_sensenova_models.sh base
#                bash user/default/scripts/restore_sensenova_models.sh info

# 3b. ComfyUI 原生音频模型:ACE-Step 1.5(音乐)+ Stable Audio 3 SFX(音效),共约 13G
#     单文件 safetensors,实体落 comfy_models/,在 models/ 下建软链
bash user/default/scripts/restore_audio_comfy_models.sh      # ace + sfx
# 或只恢复需要的: bash user/default/scripts/restore_audio_comfy_models.sh ace
#                bash user/default/scripts/restore_audio_comfy_models.sh sfx

# 4. 其它现有模型(LTX/z_image/gemma 等)挪走后 stop 也会丢,需各自从来源重新下载
#    再用 offload 脚本挪回临时盘(可选)
```

> 提示:断链检查 `find models/ -xtype l`(列出所有断掉的软链)。

## 相关脚本

- `scripts/download_wan22_weights.sh` — 下载 Wan2.2 各 task 权重到临时盘 + 建软链
- `scripts/restore_sensenova_models.sh` — stop→start 后恢复 SenseNova 基座 + Infographic(重建软链 + 重下)
- `scripts/restore_audio_comfy_models.sh` — stop→start 后恢复 ACE-Step 1.5(音乐)+ Stable Audio 3 SFX(音效)单文件权重 + 建软链(ComfyUI 原生节点用)
- `scripts/offload_models_to_nvme.sh` — 把根盘 models 大文件挪到临时盘 + 建软链
- `scripts/wan22_api.py` — 调用各 task 生视频(见脚本头注释)

## 如果想要持久化(不想每次 stop 都重下)

两个选择:
1. **扩根盘 EBS**:AWS 控制台把根卷扩到 500G+,然后
   `sudo growpart /dev/nvme0n1 1 && sudo resize2fs /dev/root`,权重放回根盘。
2. **挂一块独立 EBS 数据卷**:新建一个持久 EBS 卷挂到固定路径,把权重放上面
   (停机不丢,但持续计费)。

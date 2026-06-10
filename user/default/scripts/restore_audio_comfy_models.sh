#!/usr/bin/env bash
# stop→start 之后一键恢复「ComfyUI 原生音频模型」权重(ACE-Step 1.5 + Stable Audio 3 SFX)。
#
# 与 SenseNova 不同:这些模型走 ComfyUI 原生节点,权重是单文件 .safetensors,
# 实体放临时盘 comfy_models/,在 ComfyUI/models/ 下建同名软链(沿用现有约定)。
#   实体: /opt/dlami/nvme/comfy_models/models/{checkpoints,text_encoders}/<file>
#   软链: <ComfyUI>/models/{checkpoints,text_encoders}/<file>
#
# 临时盘是 EC2 instance store:reboot 不丢,stop→start 全部清空 → 需重跑本脚本。
#
# 模型(均为 Comfy-Org 官方打包、非 gated、可商用):
#   ACE-Step 1.5 (音乐, MIT):
#     checkpoints/ace_step_1.5_turbo_aio.safetensors   ← all-in-one(含 VAE+文本编码器,单文件即可跑)
#   Stable Audio 3 SFX (音效):
#     checkpoints/stable_audio_3_small_sfx.safetensors
#     text_encoders/t5gemma_b_b_ul2.safetensors        ← SFX 所需文本编码器
#
# 用法:
#   bash restore_audio_comfy_models.sh            # 全部(ace + sfx + ace_hq)
#   bash restore_audio_comfy_models.sh ace        # 只 ACE-Step 1.5 turbo aio(单文件,最快)
#   bash restore_audio_comfy_models.sh sfx        # 只 Stable Audio 3 SFX
#   bash restore_audio_comfy_models.sh ace_hq     # 只 ACE-Step 1.5 高质量 split 套件(xl_sft+vae+qwen)
set -euo pipefail

export HF_HUB_ENABLE_HF_TRANSFER=1
HF=/home/ubuntu/miniconda3/envs/sensenova/bin/hf
NVME=/opt/dlami/nvme
COMFY=/home/ubuntu/projects/mead/ComfyUI
MODELS_NVME=$NVME/comfy_models/models

ACE_REPO="Comfy-Org/ace_step_1.5_ComfyUI_files"
SFX_REPO="Comfy-Org/stable-audio-3"

# 1. 确认临时盘已挂载(没挂就报错退出,别往根盘写炸根盘)
if ! mountpoint -q "$NVME"; then
    echo "✗ 临时盘 $NVME 未挂载。先确认 instance store 已挂好再跑本脚本。"
    exit 1
fi

mkdir -p "$MODELS_NVME/checkpoints" "$MODELS_NVME/text_encoders" "$MODELS_NVME/diffusion_models" "$MODELS_NVME/vae"

# 下载 repo 中单个文件到临时盘对应子目录,并在 ComfyUI/models 下建软链。
#   $1 repo   $2 repo内路径(如 checkpoints/foo.safetensors)   $3 comfy子目录(checkpoints/text_encoders)
get_file() {
    local repo="$1" repo_path="$2" sub="$3"
    local fname; fname="$(basename "$repo_path")"
    local dest="$MODELS_NVME/$sub/$fname"
    local link="$COMFY/models/$sub/$fname"

    echo "  [下载/校验] $repo :: $repo_path"
    # hf download 把文件放进 HF 缓存,--local-dir 直接落到目标目录(自带断点续传/校验)
    "$HF" download "$repo" "$repo_path" --local-dir "$MODELS_NVME/.dl/$sub" >/dev/null
    # 把下载到的实际文件移动/对齐到扁平的 dest(repo_path 可能带子目录前缀)
    local dl_path="$MODELS_NVME/.dl/$sub/$repo_path"
    if [ -f "$dl_path" ] && [ "$dl_path" != "$dest" ]; then
        mv -f "$dl_path" "$dest"
    fi
    # 建/修软链
    if [ -L "$link" ] || [ ! -e "$link" ]; then
        ln -sfn "$dest" "$link"
        echo "       链 $link -> $dest"
    elif [ -e "$link" ]; then
        echo "       ⚠ $link 已是实体文件,跳过建链(请手动确认)"
    fi
}

do_ace() {
    echo "==================== ACE-Step 1.5 (音乐) ===================="
    get_file "$ACE_REPO" "checkpoints/ace_step_1.5_turbo_aio.safetensors" "checkpoints"
}

do_sfx() {
    echo "==================== Stable Audio 3 SFX (音效) ===================="
    # small_sfx: 专精音效,最省显存;medium: 音乐+音效通用,质量更高(推荐做音效用 medium)
    get_file "$SFX_REPO" "checkpoints/stable_audio_3_small_sfx.safetensors" "checkpoints"
    get_file "$SFX_REPO" "checkpoints/stable_audio_3_medium.safetensors" "checkpoints"
    get_file "$SFX_REPO" "text_encoders/t5gemma_b_b_ul2.safetensors" "text_encoders"
}

# ACE-Step 1.5 高质量 split 套件:xl_sft UNET(50步,最高质)+ VAE + 双文本编码器
# 配合 split 工作流使用,质量明显高于单文件 turbo aio。供 6 大能力的高质量路径。
do_ace_hq() {
    echo "==================== ACE-Step 1.5 高质量 split 套件 ===================="
    get_file "$ACE_REPO" "split_files/diffusion_models/acestep_v1.5_xl_sft_bf16.safetensors" "diffusion_models"
    get_file "$ACE_REPO" "split_files/vae/ace_1.5_vae.safetensors" "vae"
    get_file "$ACE_REPO" "split_files/text_encoders/qwen_0.6b_ace15.safetensors" "text_encoders"
    get_file "$ACE_REPO" "split_files/text_encoders/qwen_1.7b_ace15.safetensors" "text_encoders"
}

ALL=(ace sfx ace_hq)
TARGETS=("$@")
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=("${ALL[@]}")

for t in "${TARGETS[@]}"; do
    case "$t" in
        ace) do_ace;;
        sfx) do_sfx;;
        ace_hq) do_ace_hq;;
        *) echo "未知目标: $t (可选 ace / sfx / ace_hq)";;
    esac
done

# 清理 hf download 的中转目录(实体已 mv 到正式位置)
rm -rf "$MODELS_NVME/.dl" 2>/dev/null || true

echo "==================== 完成 ===================="
echo "临时盘:"; df -h "$NVME" | tail -1
echo "checkpoints 软链:"; ls -l "$COMFY"/models/checkpoints/*.safetensors 2>/dev/null | grep -E "ace_step|stable_audio" || true
echo "diffusion_models 软链:"; ls -l "$COMFY"/models/diffusion_models/*.safetensors 2>/dev/null | grep -E "acestep" || true
echo "vae 软链:"; ls -l "$COMFY"/models/vae/*.safetensors 2>/dev/null | grep -E "ace_1.5" || true
echo "text_encoders 软链:"; ls -l "$COMFY"/models/text_encoders/*.safetensors 2>/dev/null | grep -E "t5gemma|qwen.*ace15" || true

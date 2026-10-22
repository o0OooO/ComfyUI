#!/usr/bin/env bash
# 恢复/校验多参考图对比用的两个开源模型权重(Qwen-Image-Edit-2511 / FLUX.2-dev)。
#
# 背景:这两个模型放在 **持久 EBS 盘** /mnt/models(300G, vol-0426c78922b2dcd49),
#   跟 SenseNova / wan 走的临时盘不同 —— EBS stop→start 不会丢,所以正常情况下
#   本脚本只是"校验一遍"秒过,不需要重新下载。
#
#   对比一下三块盘:
#     /                 300G EBS   根盘,持久
#     /opt/dlami/nvme   419G 临时盘 instance store,stop→start 全部清空
#                                  (wan_weights / comfy_models / hf_cache 都在这)
#     /mnt/models       300G EBS   持久,本脚本管理的两个模型在这
#
#   所以 stop→start 之后会是"混合状态":本脚本的模型还在,但 wan/SenseNova 的软链会断,
#   那些要另外跑 restore_sensenova_models.sh / download_wan22_weights.sh。
#
# 模型(均为 Comfy-Org 重打包的 ComfyUI 单文件版,已量化,比原始 repo 省一半空间):
#   Qwen-Image-Edit-2511  fp8mixed  20.5G + Qwen2.5-VL-7B fp8 9.4G + VAE 0.25G  Apache-2.0
#   FLUX.2-dev            fp8mixed  35.5G + Mistral-3-Small fp8 18.0G + VAE 0.34G
#                                   (权重非商用许可,但生成的图可商用)
#
# 用法:
#   bash restore_multiref_models.sh          # 全部校验/恢复
#   bash restore_multiref_models.sh qwen     # 只 Qwen
#   bash restore_multiref_models.sh flux     # 只 FLUX.2
#   bash restore_multiref_models.sh flux3    # 只 FLUX.3 Klein(int8,约 20G)
set -euo pipefail

export HF_HUB_ENABLE_HF_TRANSFER=1
HF=/home/ubuntu/miniconda3/envs/sensenova/bin/hf
COMFY=/home/ubuntu/projects/mead/ComfyUI
DISK=/mnt/models
UUID=eae16a3a-dbe9-48b6-8b32-80ce24d7e686

# ---------------------------------------------------------------- 挂载检查
# EBS 盘已写进 /etc/fstab(UUID + nofail),正常自动挂载;没挂上就尝试挂一次。
if ! mountpoint -q "$DISK"; then
    echo "[挂载] $DISK 未挂载,尝试挂载 UUID=$UUID"
    sudo mkdir -p "$DISK"
    if ! sudo mount UUID="$UUID" "$DISK" 2>/dev/null; then
        echo "✗ 挂载失败。检查 EBS 卷 vol-0426c78922b2dcd49 是否还 attach 在实例上:"
        echo "    lsblk                       # 找 300G 的 Amazon Elastic Block Store"
        echo "    sudo blkid | grep $UUID"
        exit 1
    fi
    sudo chown ubuntu:ubuntu "$DISK"
    echo "  ✓ 已挂载"
fi
mkdir -p "$DISK"/{diffusion_models,text_encoders,vae,loras}

# ---------------------------------------------------------------- 下载 + 软链
# hf download 自带校验/断点续传:文件已完整则秒过。
# 下到 $DISK 后按 ComfyUI 约定软链回 models/ —— 跟 offload_models_to_nvme.sh 一个路子,
# 只是目标盘换成了持久 EBS。
get() {
    local repo="$1" remote="$2" kind="$3" name="$4"
    local tmp="$DISK/_dl/${repo//\//_}"
    local dest="$DISK/$kind/$name"
    local link="$COMFY/models/$kind/$name"

    if [ ! -f "$dest" ]; then
        echo "  [下载] $name"
        "$HF" download "$repo" "$remote" --local-dir "$tmp"
        mv "$tmp/$remote" "$dest"
    else
        echo "  [已有] $name ($(du -h "$dest" | cut -f1))"
    fi

    # 软链:断链/指错则重建
    if [ -L "$link" ] && [ "$(readlink -f "$link")" = "$dest" ]; then
        :
    elif [ -e "$link" ] && [ ! -L "$link" ]; then
        echo "  ⚠ $link 是实体文件(非软链),跳过以免误删"
    else
        ln -sfn "$dest" "$link"
        echo "         软链 -> models/$kind/$name"
    fi
}

do_qwen() {
    echo "==================== Qwen-Image-Edit-2511 ===================="
    local R1=Comfy-Org/Qwen-Image-Edit_ComfyUI R2=Comfy-Org/Qwen-Image_ComfyUI
    get "$R1" split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors \
        diffusion_models qwen_image_edit_2511_fp8mixed.safetensors
    get "$R2" split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors \
        text_encoders qwen_2.5_vl_7b_fp8_scaled.safetensors
    get "$R2" split_files/vae/qwen_image_vae.safetensors \
        vae qwen_image_vae.safetensors
}

do_flux() {
    echo "==================== FLUX.2-dev ===================="
    local R=Comfy-Org/flux2-dev
    get "$R" split_files/diffusion_models/flux2_dev_fp8mixed.safetensors \
        diffusion_models flux2_dev_fp8mixed.safetensors
    get "$R" split_files/text_encoders/mistral_3_small_flux2_fp8.safetensors \
        text_encoders mistral_3_small_flux2_fp8.safetensors
    get "$R" split_files/vae/flux2-vae.safetensors \
        vae flux2-vae.safetensors
    # Turbo LoRA:4~8 步出图,调 prompt 阶段省时间(可选)
    get "$R" split_files/loras/Flux2TurboComfyv2.safetensors \
        loras Flux2TurboComfyv2.safetensors
}

do_flux3() {
    echo "==================== FLUX.3 Image Klein ===================="
    # gated:需先在 HF 页面同意 FLUX Kommunity License。VAE 与 FLUX.2 共用 flux2-vae。
    # int8 而不是 nvfp4:L40S(sm_89) 没有原生 FP4。
    local R=black-forest-labs/flux-3-image-klein-comfy
    get "$R" diffusion_models/flux3-image-klein-int8-convrot.safetensors \
        diffusion_models flux3-image-klein-int8-convrot.safetensors
    get "$R" text_encoders/qwen3vl_4b_int8_convrot.safetensors \
        text_encoders qwen3vl_4b_int8_convrot.safetensors
    # prompt upsampler(Qwen3.5-9B):官方要求短 prompt 先扩写成完整画面描述
    get "$R" text_encoders/qwen3.5_9b_flux3_prompt_upsampler_int8_convrot.safetensors \
        text_encoders qwen3.5_9b_flux3_prompt_upsampler_int8_convrot.safetensors
    get "$R" vae/flux2-vae.safetensors \
        vae flux2-vae.safetensors
}

TARGETS=("$@")
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=(qwen flux flux3)
for t in "${TARGETS[@]}"; do
    case "$t" in
        qwen) do_qwen;;
        flux) do_flux;;
        flux3) do_flux3;;
        *) echo "未知目标: $t (可选 qwen / flux / flux3)";;
    esac
done

rm -rf "$DISK/_dl"

echo "==================== 完成 ===================="
echo "EBS 盘:"; df -h "$DISK" | tail -1
echo "软链检查(断链会显示 BROKEN):"
for f in diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors \
         text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors \
         vae/qwen_image_vae.safetensors \
         diffusion_models/flux2_dev_fp8mixed.safetensors \
         text_encoders/mistral_3_small_flux2_fp8.safetensors \
         vae/flux2-vae.safetensors \
         loras/Flux2TurboComfyv2.safetensors \
         diffusion_models/flux3-image-klein-int8-convrot.safetensors \
         text_encoders/qwen3vl_4b_int8_convrot.safetensors \
         text_encoders/qwen3.5_9b_flux3_prompt_upsampler_int8_convrot.safetensors; do
    p="$COMFY/models/$f"
    if [ -f "$p" ]; then printf "  OK      %s\n" "$f"
    elif [ -L "$p" ]; then printf "  BROKEN  %s\n" "$f"
    else printf "  MISSING %s\n" "$f"; fi
done

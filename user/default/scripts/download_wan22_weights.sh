#!/usr/bin/env bash
# 下载 Wan2.2 各 task 所需权重到临时盘,并软链进 ComfyUI models 目录。
#
# 权重存放:/opt/dlami/nvme/wan_weights(instance store 临时盘,391G 可用)
#   ⚠️ 实例 stop/terminate 后会丢失,需重新运行本脚本下载(reboot 不丢)。
# 软链目标:ComfyUI/models/{diffusion_models,audio_encoders}
#
# 文件名/来源均已核实(HuggingFace,带精确字节)。fp8 量化版优先以省空间。
# 用法:
#   bash download_wan22_weights.sh            # 下全部 task
#   bash download_wan22_weights.sh t2v s2v    # 只下指定 task
set -euo pipefail

export HF_HUB_ENABLE_HF_TRANSFER=1
HF=/home/ubuntu/miniconda3/envs/sensenova/bin/hf
COMFY=/home/ubuntu/projects/mead/ComfyUI
STORE=/opt/dlami/nvme/wan_weights
DM=$STORE/diffusion_models
AE=$STORE/audio_encoders
mkdir -p "$DM" "$AE" "$COMFY/models/diffusion_models" "$COMFY/models/audio_encoders"

COMFY_REPO="Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
KIJAI="Kijai/WanVideo_comfy"
KIJAI_FP8="Kijai/WanVideo_comfy_fp8_scaled"

# 下载一个文件到 $DM 并软链进 models/diffusion_models/<目标名>
# 参数: <repo> <repo内路径> <软链目标文件名>
get_dm() {
    local repo="$1" path="$2" link="$3"
    local fname; fname=$(basename "$path")
    if [ -f "$DM/$fname" ]; then
        echo "[skip] 已存在 $fname"
    else
        echo "[下载] $repo : $path"
        $HF download "$repo" "$path" --local-dir "$STORE/_dl"
        mv "$STORE/_dl/$path" "$DM/$fname"
    fi
    ln -sf "$DM/$fname" "$COMFY/models/diffusion_models/$link"
    echo "       软链 -> models/diffusion_models/$link"
}

get_audio() {
    local repo="$1" path="$2"; local fname; fname=$(basename "$path")
    if [ -f "$AE/$fname" ]; then echo "[skip] 已存在 $fname"; else
        echo "[下载] $repo : $path"; $HF download "$repo" "$path" --local-dir "$STORE/_dl"
        mv "$STORE/_dl/$path" "$AE/$fname"
    fi
    ln -sf "$AE/$fname" "$COMFY/models/audio_encoders/$(basename "$path")"
    echo "       软链 -> models/audio_encoders/$(basename "$path")"
}

dl_t2v() {
    get_dm "$COMFY_REPO" "split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors" "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"
    get_dm "$COMFY_REPO" "split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors" "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"
}
dl_fun_inpaint() {
    get_dm "$COMFY_REPO" "split_files/diffusion_models/wan2.2_fun_inpaint_high_noise_14B_fp8_scaled.safetensors" "wan2.2_fun_inpaint_high_noise_14B_fp8_scaled.safetensors"
    get_dm "$COMFY_REPO" "split_files/diffusion_models/wan2.2_fun_inpaint_low_noise_14B_fp8_scaled.safetensors" "wan2.2_fun_inpaint_low_noise_14B_fp8_scaled.safetensors"
}
dl_fun_control() {
    get_dm "$COMFY_REPO" "split_files/diffusion_models/wan2.2_fun_control_high_noise_14B_fp8_scaled.safetensors" "wan2.2_fun_control_high_noise_14B_fp8_scaled.safetensors"
    get_dm "$COMFY_REPO" "split_files/diffusion_models/wan2.2_fun_control_low_noise_14B_fp8_scaled.safetensors" "wan2.2_fun_control_low_noise_14B_fp8_scaled.safetensors"
}
dl_s2v() {
    get_dm "$COMFY_REPO" "split_files/diffusion_models/wan2.2_s2v_14B_fp8_scaled.safetensors" "wan2.2_s2v_14B_fp8_scaled.safetensors"
    get_audio "$COMFY_REPO" "split_files/audio_encoders/wav2vec2_large_english_fp16.safetensors"
}
dl_vace() {
    # Comfy-Org 只有 fp16 完整版(单文件,最稳)
    get_dm "Comfy-Org/Wan_2.1_ComfyUI_Repackaged" "split_files/diffusion_models/wan2.1_vace_14B_fp16.safetensors" "wan2.1_vace_14B_fp16.safetensors"
}
dl_phantom() {
    # Comfy-Org 无 phantom,用 Kijai fp8
    get_dm "$KIJAI" "Phantom-Wan-14B_fp8_e4m3fn.safetensors" "wan2.1_phantom_14B_fp8.safetensors"
}
dl_animate() {
    # Comfy-Org 只有 bf16,用 Kijai fp8 v2
    get_dm "$KIJAI_FP8" "Wan22Animate/Wan2_2-Animate-14B_fp8_scaled_e4m3fn_KJ_v2.safetensors" "wan2.2_animate_14B_fp8_scaled.safetensors"
}

ALL=(t2v fun_inpaint fun_control s2v vace phantom animate)
TARGETS=("$@")
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=("${ALL[@]}")

for t in "${TARGETS[@]}"; do
    echo "==================== $t ===================="
    case "$t" in
        t2v) dl_t2v;; fun_inpaint) dl_fun_inpaint;; fun_control) dl_fun_control;;
        s2v) dl_s2v;; vace) dl_vace;; phantom) dl_phantom;; animate) dl_animate;;
        *) echo "未知 task: $t";;
    esac
done
rm -rf "$STORE/_dl"
echo "==================== 完成 ===================="
df -h /opt/dlami/nvme | tail -1

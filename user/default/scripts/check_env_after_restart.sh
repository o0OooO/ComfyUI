#!/usr/bin/env bash
# 重启(尤其 stop→start)之后先跑这个:体检一遍,并直接告诉你缺什么、该跑哪条命令。
#
# 为什么需要它:本机是"多块盘混用",持久性各不相同,重启后是**混合状态** ——
# 一部分模型还在,一部分全丢了,光看 ComfyUI 界面不容易分辨(断链模型只是从
# 下拉列表里消失,不报错)。
#
#   /                 EBS   持久   根盘,代码 + 自定义节点
#   /mnt/models       EBS   持久   Qwen-Image-Edit-2511 / FLUX.2-dev
#   /opt/dlami/nvme   本地  stop→start **全部清空**   SenseNova / wan / 音频模型
#
# 只读脚本,不会下载或修改任何东西。
#
# 用法: bash user/default/scripts/check_env_after_restart.sh
set -uo pipefail

COMFY=/home/ubuntu/projects/mead/ComfyUI
EBS=/mnt/models
NVME=/opt/dlami/nvme
EBS_UUID=eae16a3a-dbe9-48b6-8b32-80ce24d7e686

ok=0; bad=0
TODO=()
pass() { printf "  \033[32m✓\033[0m %s\n" "$1"; ok=$((ok+1)); }
fail() { printf "  \033[31m✗\033[0m %s\n" "$1"; bad=$((bad+1)); }
note() { printf "  \033[33m!\033[0m %s\n" "$1"; }

cd "$COMFY" || exit 1

echo "==================== 1. 挂载 ===================="
if mountpoint -q "$EBS"; then
    pass "$EBS 已挂载($(df -h "$EBS" | awk 'NR==2{print $4" 可用"}'))"
else
    fail "$EBS 未挂载(Qwen/FLUX 权重都在这)"
    TODO+=("sudo mount UUID=$EBS_UUID $EBS   # 或直接跑 restore_multiref_models.sh,它会自己挂")
fi
if mountpoint -q "$NVME"; then
    pass "$NVME 已挂载($(df -h "$NVME" | awk 'NR==2{print $4" 可用"}'))"
else
    fail "$NVME 未挂载(临时盘,DLAMI 一般自动挂)"
    TODO+=("检查临时盘挂载:lsblk;必要时 sudo mount /dev/nvme1n1 $NVME")
fi

echo
echo "==================== 2. 自定义节点(子模块) ===================="
# SenseNova 的三个节点全靠这个子模块;它在根盘所以 stop 不丢,
# 但换机器 / 重新 clone 时必须 submodule update,否则工作流报"节点不存在"。
if [ -f custom_nodes/ComfyUI-SenseNova-U1/nodes.py ]; then
    pass "ComfyUI-SenseNova-U1 存在($(git -C custom_nodes/ComfyUI-SenseNova-U1 describe --tags --always 2>/dev/null || echo '?'))"
else
    fail "ComfyUI-SenseNova-U1 缺失 —— SenseNova 所有节点都不可用"
    TODO+=("git submodule update --init --recursive")
fi
if [ -f custom_nodes/ComfyUI-Video-Depth-Anything/node.py ]; then
    pass "ComfyUI-Video-Depth-Anything 存在"
else
    fail "ComfyUI-Video-Depth-Anything 缺失 —— 视频深度工作流不可用"
    TODO+=("git submodule update --init --recursive")
fi

echo
echo "==================== 2b. Video-Depth-Anything(依赖 + EBS 权重) ===================="
# 依赖装在 sensenova conda 环境(根盘),权重在 /mnt/models,models/ 下只是软链
if /home/ubuntu/miniconda3/envs/sensenova/bin/python -c "import cv2, matplotlib, easydict, imageio, OpenEXR" 2>/dev/null; then
    pass "VDA pip 依赖齐全"
else
    fail "VDA pip 依赖缺失"
    TODO+=("~/miniconda3/envs/sensenova/bin/pip install -r custom_nodes/ComfyUI-Video-Depth-Anything/requirements.txt")
fi
if [ -f models/videodepthanything/video_depth_anything_vitl.pth ]; then
    pass "videodepthanything/video_depth_anything_vitl.pth"
else
    fail "videodepthanything/video_depth_anything_vitl.pth(软链或权重缺失)"
    TODO+=("ln -sfn $EBS/videodepthanything models/videodepthanything   # 权重本身也没了的话,节点首次运行会自动从 HF 下载")
fi

echo
echo "==================== 3. Qwen / FLUX 权重(EBS,应当保留) ===================="
miss_multiref=0
for f in diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors \
         text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors \
         vae/qwen_image_vae.safetensors \
         diffusion_models/flux2_dev_fp8mixed.safetensors \
         text_encoders/mistral_3_small_flux2_fp8.safetensors \
         vae/flux2-vae.safetensors \
         loras/Flux2TurboComfyv2.safetensors; do
    if [ -f "models/$f" ]; then pass "$f"
    else fail "$f"; miss_multiref=1; fi
done
[ $miss_multiref -eq 1 ] && TODO+=("bash user/default/scripts/restore_multiref_models.sh")

echo
echo "==================== 4. SenseNova 权重(临时盘,stop 后必丢) ===================="
HFC=$(readlink -f /home/ubuntu/.cache/huggingface 2>/dev/null || echo "")
if [ -z "$HFC" ] || [ ! -d "$HFC" ]; then
    fail "~/.cache/huggingface 软链断了或不存在"
    TODO+=("bash user/default/scripts/restore_sensenova_models.sh")
else
    pass "HF 缓存 -> $HFC"
    miss_sn=0
    for m in SenseNova-U1-8B-MoT SenseNova-U1-8B-MoT-Infographic; do
        d="$HFC/hub/models--sensenova--$m"
        # 只看目录不够 —— 丢盘后可能剩个空壳,所以按体积判断(完整约 33G)
        sz=$(du -sm "$d" 2>/dev/null | cut -f1 || echo 0)
        if [ "${sz:-0}" -gt 20000 ]; then pass "$m ($((sz/1024))G)"
        else fail "$m 缺失或不完整 (${sz:-0}MB)"; miss_sn=1; fi
    done
    [ $miss_sn -eq 1 ] && TODO+=("bash user/default/scripts/restore_sensenova_models.sh")
fi

echo
echo "==================== 5. LTX-2.3 权重(4 个在临时盘,stop 后必丢) ===================="
# 基座/文本编码器/上采样/蒸馏 LoRA 都在 /opt/dlami/nvme,约 40G,stop→start 全没。
# Ingredients IC-LoRA 特意放了持久 EBS /mnt/models,正常不会丢。
miss_ltx=0
for f in checkpoints/ltx-2.3-22b-dev-fp8.safetensors \
         text_encoders/gemma_3_12B_it_fp4_mixed.safetensors \
         latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
         loras/ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors; do
    if [ -f "models/$f" ]; then pass "$f"
    else fail "$f"; miss_ltx=1; fi
done
# IC-LoRA 是 gated 仓库,缺了单独提示(要 token,不是重下就行)
if [ -f "models/loras/ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors" ]; then
    pass "loras/ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors (多参考用)"
else
    note "Ingredients IC-LoRA 缺失 —— 多参考生视频跑不了。"
    note "  它是 HF gated 仓库:先网页点 Agree and Access,再 hf auth login"
    note "  https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients"
    miss_ltx=1
fi
[ $miss_ltx -eq 1 ] && TODO+=("bash user/default/scripts/restore_ltx23_models.sh")

echo
echo "==================== 6. 断链的软链 ===================="
broken=$(find models/ -xtype l 2>/dev/null)
if [ -z "$broken" ]; then
    pass "没有断链"
else
    n=$(echo "$broken" | wc -l)
    note "$n 个断链(ComfyUI 不会崩,这些模型只是从下拉列表消失):"
    echo "$broken" | sed 's/^/      /' | head -15
    [ "$n" -gt 15 ] && echo "      ... 还有 $((n-15)) 个"
    note "wan / 音频模型的断链分别跑 download_wan22_weights.sh / restore_audio_comfy_models.sh"
fi

echo
echo "==================== 7. ComfyUI 服务 ===================="
if curl -s -m 5 http://127.0.0.1:8188/system_stats >/dev/null 2>&1; then
    pass "ComfyUI 在线 (127.0.0.1:8188)"
else
    note "ComfyUI 未响应 —— 权重恢复完之后再启动它(启动时才扫描模型列表):"
    note "  bash user/default/scripts/start_comfy.sh"
fi
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null \
    | sed 's/^/  显存: /' || note "nvidia-smi 不可用"

echo
echo "======================================================="
printf "正常 %d 项,异常 %d 项\n" "$ok" "$bad"
if [ ${#TODO[@]} -eq 0 ]; then
    echo "环境完整,可以直接用。"
else
    echo
    echo "需要执行(按顺序):"
    printf '  %s\n' "${TODO[@]}"
    echo
    echo "提示:恢复完权重后重启 ComfyUI,它只在启动时扫描 models/ 目录。"
fi

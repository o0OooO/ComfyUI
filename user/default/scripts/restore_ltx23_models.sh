#!/usr/bin/env bash
# 恢复/校验 LTX-2.3 多参考生视频(Ingredients IC-LoRA)所需的全部权重。
#
# ⚠ 为什么必须有这个脚本:
#   现有 4 个 LTX-2.3 权重(约 40G)当初全下到了 **临时盘** /opt/dlami/nvme,
#   stop→start 会被整盘清空 —— 也就是重启后 LTX 工作流会全线报"模型找不到"。
#   本脚本把它们重新拉回来;新增的 Ingredients IC-LoRA 则放 **持久 EBS** /mnt/models,
#   下次 stop→start 不用重下。
#
#   三块盘(细节见 user/default/docs/NVME_STORAGE_NOTES.md):
#     /                 300G EBS   根盘,持久(只剩 ~34G,别往这塞大模型)
#     /opt/dlami/nvme   419G 临时  stop→start 全清空  ← 大文件默认落这
#     /mnt/models       300G EBS   持久,余量充足     ← IC-LoRA 落这
#
# ⚠ Ingredients IC-LoRA 是 HF **gated** 仓库,必须先满足两件事:
#   1. 浏览器打开下面地址点 "Agree and Access"(一次性):
#        https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients
#   2. 本机有 token:  hf auth login       (或 export HF_TOKEN=hf_xxx)
#   没 token 的话本脚本会跳过它,其余 4 个照常恢复。
#
# 用法:
#   bash restore_ltx23_models.sh          # 全部校验/恢复
#   bash restore_ltx23_models.sh base     # 只基座 + 蒸馏 LoRA + 文本编码器 + 上采样
#   bash restore_ltx23_models.sh iclora   # 只 Ingredients IC-LoRA
set -euo pipefail

export HF_HUB_ENABLE_HF_TRANSFER=1
HF=/home/ubuntu/miniconda3/envs/sensenova/bin/hf
COMFY=/home/ubuntu/projects/mead/ComfyUI
NVME=/opt/dlami/nvme/comfy_models/models     # 临时盘:大基座放这
EBS=/mnt/models                              # 持久盘:IC-LoRA 放这

# ---------------------------------------------------------------- 通用取件函数
# $1 repo  $2 仓库内路径  $3 kind(models/ 下的子目录)  $4 落地文件名  $5 目标盘根
get() {
    local repo="$1" remote="$2" kind="$3" name="$4" root="$5"
    local dest="$root/$kind/$name"
    local link="$COMFY/models/$kind/$name"
    local tmp="$root/_dl/${repo//\//_}"

    mkdir -p "$root/$kind"
    if [ ! -f "$dest" ]; then
        echo "  [下载] $name  ($repo)"
        "$HF" download "$repo" "$remote" --local-dir "$tmp"
        mv "$tmp/$remote" "$dest"
    else
        echo "  [已有] $name ($(du -h "$dest" | cut -f1))"
    fi

    if [ -L "$link" ] && [ "$(readlink -f "$link")" = "$dest" ]; then
        :
    elif [ -e "$link" ] && [ ! -L "$link" ]; then
        echo "  ⚠ $link 是实体文件(非软链),跳过以免误删"
    else
        mkdir -p "$(dirname "$link")"
        ln -sfn "$dest" "$link"
        echo "         软链 -> models/$kind/$name"
    fi
}

# ---------------------------------------------------------------- 基座四件套
do_base() {
    echo "==================== LTX-2.3 基座(临时盘 $NVME)===================="
    if ! mountpoint -q /opt/dlami/nvme; then
        echo "✗ /opt/dlami/nvme 没挂载,先处理临时盘再来。"; return 1
    fi
    # 基座 22B fp8(28G)。注意在 Lightricks/LTX-2.3-fp8 仓库根目录,
    # 不在 Comfy-Org/ltx-2.3(那边只有 loras/upscaler,没有 fp8 基座)。
    get Lightricks/LTX-2.3-fp8 ltx-2.3-22b-dev-fp8.safetensors \
        checkpoints ltx-2.3-22b-dev-fp8.safetensors "$NVME"
    # Gemma-3-12B 文本编码器 fp4(LTX-2/2.3 通用)
    get Comfy-Org/ltx-2 split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors \
        text_encoders gemma_3_12B_it_fp4_mixed.safetensors "$NVME"
    # 空间上采样器 x2(两阶段高清流程用;多参考单阶段流程用不到,留着不占多少)
    get Lightricks/LTX-2.3 ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
        latent_upscale_models ltx-2.3-spatial-upscaler-x2-1.1.safetensors "$NVME"
    # 蒸馏 LoRA:8 步档就靠它。注意本机用的是 dynamic-rank 版,
    # 跟官方 blueprint 里写的 ltx-2.3-22b-distilled-lora-384-1.1 不是同一个文件。
    get Comfy-Org/ltx-2.3 \
        split_files/loras/ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors \
        loras ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors "$NVME"
    rm -rf "$NVME/_dl"
}

# ---------------------------------------------------------------- Ingredients IC-LoRA
do_iclora() {
    echo "==================== Ingredients IC-LoRA(持久盘 $EBS)===================="
    if ! mountpoint -q "$EBS"; then
        echo "  [挂载] $EBS 未挂载,尝试挂载"
        sudo mkdir -p "$EBS"
        sudo mount UUID=eae16a3a-dbe9-48b6-8b32-80ce24d7e686 "$EBS" || {
            echo "  ✗ 挂载失败,详见 restore_multiref_models.sh 里的排查步骤"; return 1; }
        sudo chown ubuntu:ubuntu "$EBS"
    fi

    # gated 仓库:没 token 直接跳过,不要让整个脚本挂掉。
    # 注意 `hf auth whoami` 未登录时也 exit 0(只是打印 "Not logged in"),
    # 所以只能判输出内容,不能判返回码。
    local who=""
    [ -z "${HF_TOKEN:-}" ] && who="$("$HF" auth whoami 2>/dev/null || true)"
    if [ -z "${HF_TOKEN:-}" ] && { [ -z "$who" ] || [ "$who" = "Not logged in" ]; }; then
        cat <<'EOF'
  ⚠ 跳过:没检测到 HF token,gated 仓库拉不动。
    补齐办法(二选一):
      hf auth login                  # 交互式贴 token
      export HF_TOKEN=hf_xxxxx       # 临时环境变量
    并确认已在网页点过 Agree and Access:
      https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients
EOF
        return 0
    fi

    get Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients \
        ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors \
        loras ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors "$EBS"
    rm -rf "$EBS/_dl"
}

TARGETS=("$@")
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=(base iclora)
for t in "${TARGETS[@]}"; do
    case "$t" in
        base)   do_base;;
        iclora) do_iclora;;
        *) echo "未知目标: $t (可选 base / iclora)";;
    esac
done

# ---------------------------------------------------------------- 收尾校验
echo "==================== 完成 ===================="
echo "磁盘:"
df -h /opt/dlami/nvme "$EBS" 2>/dev/null | grep -v ^Filesystem || true
echo "软链检查(BROKEN = 目标文件不在了,多半是临时盘被清空):"
for f in checkpoints/ltx-2.3-22b-dev-fp8.safetensors \
         text_encoders/gemma_3_12B_it_fp4_mixed.safetensors \
         latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
         loras/ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors \
         loras/ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors; do
    p="$COMFY/models/$f"
    if [ -f "$p" ]; then printf "  OK      %s\n" "$f"
    elif [ -L "$p" ]; then printf "  BROKEN  %s\n" "$f"
    else printf "  MISSING %s\n" "$f"; fi
done
cat <<'EOF'

工作流:user/default/workflows/video_ltx2_3_multiref_ingredients.json
说明书:user/default/docs/LTX23_MULTIREF_GUIDE.md
EOF

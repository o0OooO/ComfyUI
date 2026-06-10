#!/usr/bin/env bash
# stop→start 之后一键恢复 SenseNova-U1 模型权重。
#
# 背景:HF 缓存软链到临时盘 ~/.cache/huggingface -> /opt/dlami/nvme/hf_cache/huggingface
#   临时盘是 EC2 instance store:reboot 不丢,stop→start 全部清空。
#   所以每次 stop→start 后两个模型(各约 33G)都没了,本脚本重建软链 + 重新下载。
#
# 模型:
#   sensenova/SenseNova-U1-8B-MoT              基座(通用文生图 / 图片编辑)
#   sensenova/SenseNova-U1-8B-MoT-Infographic  信息图 / 海报专精版
#
# 用法:
#   bash restore_sensenova_models.sh            # 恢复两个
#   bash restore_sensenova_models.sh base       # 只恢复基座
#   bash restore_sensenova_models.sh info        # 只恢复 Infographic
set -euo pipefail

export HF_HUB_ENABLE_HF_TRANSFER=1
HF=/home/ubuntu/miniconda3/envs/sensenova/bin/hf
NVME=/opt/dlami/nvme
HF_CACHE=$NVME/hf_cache/huggingface
LINK=/home/ubuntu/.cache/huggingface

BASE_REPO="sensenova/SenseNova-U1-8B-MoT"
INFO_REPO="sensenova/SenseNova-U1-8B-MoT-Infographic"

# 1. 确认临时盘已挂载(DLAMI 通常自动挂;没挂就报错退出,别往根盘写炸根盘)
if ! mountpoint -q "$NVME"; then
    echo "✗ 临时盘 $NVME 未挂载。先确认 instance store 已挂好再跑本脚本。"
    echo "  (lsblk 看一下 nvme1n1 / vg.01-lv_ephemeral 的挂载点)"
    exit 1
fi

# 2. 重建 HF 缓存目录 + 软链(临时盘清空后软链会断)
mkdir -p "$HF_CACHE"
if [ -L "$LINK" ]; then
    # 软链存在:断链则修,指对了就留
    if [ "$(readlink -f "$LINK")" != "$HF_CACHE" ]; then
        echo "[修复] $LINK 断链/指错 -> 重指向 $HF_CACHE"
        ln -sfn "$HF_CACHE" "$LINK"
    fi
elif [ -e "$LINK" ]; then
    echo "✗ $LINK 是实体目录(非软链)。请手动处理,避免误删数据。"
    exit 1
else
    mkdir -p "$(dirname "$LINK")"
    ln -s "$HF_CACHE" "$LINK"
    echo "[建链] $LINK -> $HF_CACHE"
fi

# 3. 下载一个模型仓库(已完整则秒过 —— hf download 自带校验/断点续传)
get_model() {
    local repo="$1" name="$2"
    echo "==================== $name ===================="
    echo "[下载/校验] $repo"
    "$HF" download "$repo"
    echo "       OK -> $HF_CACHE/hub/models--${repo//\//--}"
}

ALL=(base info)
TARGETS=("$@")
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=("${ALL[@]}")

for t in "${TARGETS[@]}"; do
    case "$t" in
        base) get_model "$BASE_REPO" "基座 SenseNova-U1-8B-MoT";;
        info) get_model "$INFO_REPO" "信息图 SenseNova-U1-8B-MoT-Infographic";;
        *) echo "未知目标: $t (可选 base / info)";;
    esac
done

echo "==================== 完成 ===================="
echo "临时盘:"; df -h "$NVME" | tail -1
echo "已缓存的 SenseNova 模型:"
ls -d "$HF_CACHE"/hub/models--sensenova--* 2>/dev/null || echo "  (无)"

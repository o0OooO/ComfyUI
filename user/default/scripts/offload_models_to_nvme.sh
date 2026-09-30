#!/usr/bin/env bash
# 把 ComfyUI/models 下的大文件挪到临时盘(instance store),再软链回原位,给根盘腾空间。
#
# 临时盘:/opt/dlami/nvme/comfy_models(391G,实例 stop 后清空)
# ⚠️ 挪走的文件 stop/terminate 后会丢,需重新下载。reboot 不影响。
#
# 幂等:已是软链的跳过;已在临时盘的不重复挪。
# 用法:bash offload_models_to_nvme.sh [最小体积MB,默认500]
set -euo pipefail

COMFY=/home/ubuntu/projects/mead/ComfyUI
DEST=/opt/dlami/nvme/comfy_models
MIN_MB="${1:-500}"
mkdir -p "$DEST"

moved=0; skipped=0
while IFS= read -r f; do
    # f 形如 models/checkpoints/xxx.safetensors
    rel="${f#"$COMFY"/}"
    target="$DEST/$rel"
    if [ -L "$f" ]; then skipped=$((skipped+1)); continue; fi   # 已是软链
    mkdir -p "$(dirname "$target")"
    echo "[挪] $rel ($(du -h "$f" | cut -f1))"
    mv "$f" "$target"
    ln -s "$target" "$f"
    moved=$((moved+1))
done < <(find "$COMFY/models" -type f -size +"${MIN_MB}"M 2>/dev/null)

echo "----------------------------------------"
echo "挪动 $moved 个文件,跳过(已软链) $skipped 个"
echo "根盘:"; df -h / | tail -1
echo "临时盘:"; df -h /opt/dlami/nvme | tail -1

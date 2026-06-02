#!/usr/bin/env bash
# SenseNova-U1 ComfyUI 容器入口:启动前确保模型权重就位,再拉起 ComfyUI。
#
# 权重(约 33GB)不打进镜像。强烈建议运行时挂载 HF 缓存卷:
#   -v $HOME/.cache/huggingface:/root/.cache/huggingface
# 这样权重只需下载一次,后续容器复用,不会每次重下。
set -euo pipefail

MODEL_ID="${SENSENOVA_MODEL_ID:-sensenova/SenseNova-U1-8B-MoT}"
# 是否在启动时自动下载缺失的权重(默认开)。设 SKIP_MODEL_DOWNLOAD=1 可跳过。
SKIP_DOWNLOAD="${SKIP_MODEL_DOWNLOAD:-0}"

# HF 用 HF_HOME 定位缓存;模型快照落在 $HF_HOME/hub/models--<org>--<name>/snapshots/
HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
model_dir_glob="${HF_HOME}/hub/models--${MODEL_ID//\//--}/snapshots"

have_weights() {
    # 判定:存在快照目录且其中至少有一个 .safetensors(顺软链解析)
    compgen -G "${model_dir_glob}/*/*.safetensors" > /dev/null 2>&1
}

if [ "${SKIP_DOWNLOAD}" != "1" ]; then
    if have_weights; then
        echo "[entrypoint] 已检测到模型权重缓存: ${MODEL_ID}"
    else
        if [ ! -d "${HF_HOME}" ] || [ ! -w "${HF_HOME}" ]; then
            echo "[entrypoint][警告] HF 缓存目录 ${HF_HOME} 不存在或不可写。"
            echo "[entrypoint][警告] 建议挂载卷: -v \$HOME/.cache/huggingface:${HF_HOME}"
        fi
        echo "[entrypoint] 未找到权重,开始下载 ${MODEL_ID}(约 33GB,首次较慢)..."
        echo "[entrypoint] 提示:挂载 HF 缓存卷可避免每次重新下载。"
        huggingface-cli download "${MODEL_ID}" --quiet || \
        hf download "${MODEL_ID}"
        echo "[entrypoint] 权重下载完成。"
    fi
else
    echo "[entrypoint] SKIP_MODEL_DOWNLOAD=1,跳过权重检查(假定运行时已就位)。"
fi

echo "[entrypoint] 启动 ComfyUI..."
# 透传 docker run / compose 传入的额外参数(没有则用默认 listen)
if [ "$#" -gt 0 ]; then
    exec python main.py "$@"
else
    exec python main.py --listen 0.0.0.0 --port 8188
fi

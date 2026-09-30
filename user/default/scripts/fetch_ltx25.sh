#!/usr/bin/env bash
# 下载 LTX-2.5 生视频所需的全部权重(约 41G,全部落 **持久盘** /mnt/models)。
#
# 和 LTX-2.3 的区别(为什么不能沿用 restore_ltx23_models.sh):
#   2.3 是「一个 all-in-one checkpoint + LTXAVTextEncoderLoader/LTXVAudioVAELoader」;
#   2.5 拆成了 4 类独立文件,用标准核心节点加载:
#     diffusion_models/  UNETLoader
#     text_encoders/     CLIPLoader(type=ltxv)
#     vae/               VAELoader  ×2(视频 VAE + 音频 VAE)
#     latent_upscale_models/  LatentUpscaleModelLoader(两阶段高清的第二阶段用)
#   另外 2.3 的权重当初全下在临时盘 /opt/dlami/nvme(stop 就没了);
#   2.5 一律放 /mnt/models(EBS 持久盘,已扩到 442G)。
#
# ⚠ Lightricks/LTX-2.5 和 Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients 都是 HF **gated**
#   (gated: auto —— 仓库元数据公开可读,但下文件会 401)。必须先做两件事:
#     1. 浏览器里各点一次 "Agree and Access":
#          https://huggingface.co/Lightricks/LTX-2.5
#          https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients
#     2. 本机登录:  hf auth login        (或 export HF_TOKEN=hf_xxx)
#   没满足就跑本脚本会在 precheck 阶段直接告诉你差哪一步,不会下到一半才炸。
#
# 用法:
#   bash fetch_ltx25.sh              # 必需的 6 个文件(基座+TE+双VAE+上采样+IC-LoRA)
#   bash fetch_ltx25.sh base         # 只基座 4 件套 + 上采样
#   bash fetch_ltx25.sh iclora       # 只 Ingredients IC-LoRA(多参考工作流用)
#   bash fetch_ltx25.sh enhancer     # 可选:提示词增强器 gemma4-e2b(5.2G,不 gated)
#   bash fetch_ltx25.sh check        # 只体检,不下载
set -euo pipefail

export HF_HUB_ENABLE_HF_TRANSFER=1
HF=/home/ubuntu/miniconda3/envs/sensenova/bin/hf
COMFY=/home/ubuntu/projects/mead/ComfyUI
EBS=/mnt/models                               # 持久盘:2.5 全部放这

REPO_MAIN=Lightricks/LTX-2.5
REPO_ICL=Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients
REPO_ENH=Comfy-Org/gemma-4

# 清单:<repo> <仓库内路径> <models/ 下的子目录> <落地文件名>
# 注意 2.5 仓库里的路径本身带子目录,和 models/ 下的子目录同名但要分开写。
MANIFEST_BASE=(
  "$REPO_MAIN|diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors|diffusion_models|ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
  "$REPO_MAIN|text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors|text_encoders|gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
  "$REPO_MAIN|vae/ltx-2.5-video-vae-bf16.safetensors|vae|ltx-2.5-video-vae-bf16.safetensors"
  "$REPO_MAIN|vae/ltx-2.5-audio-vae-bf16.safetensors|vae|ltx-2.5-audio-vae-bf16.safetensors"
  "$REPO_MAIN|latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors|latent_upscale_models|ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
)
MANIFEST_ICL=(
  "$REPO_ICL|ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors|loras|ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors"
)
MANIFEST_ENH=(
  "$REPO_ENH|text_encoders/gemma4_e2b_it_int8_convrot.safetensors|text_encoders|gemma4_e2b_it_int8_convrot.safetensors"
)

# ---------------------------------------------------------------- 取件函数
# 与 restore_ltx23_models.sh 的 get() 同款:先下到 _dl 再 mv,最后软链回 models/
get() {
    local repo="$1" remote="$2" kind="$3" name="$4"
    local dest="$EBS/$kind/$name"
    local link="$COMFY/models/$kind/$name"
    local tmp="$EBS/_dl/${repo//\//_}"

    mkdir -p "$EBS/$kind"
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

run_manifest() {
    local -n arr="$1"
    local row repo remote kind name
    for row in "${arr[@]}"; do
        IFS='|' read -r repo remote kind name <<<"$row"
        get "$repo" "$remote" "$kind" "$name"
    done
    rm -rf "$EBS/_dl"
}

# ---------------------------------------------------------------- 前置体检
hf_token() {
    [ -n "${HF_TOKEN:-}" ] && { echo "$HF_TOKEN"; return; }
    "$HF" auth token 2>/dev/null || true
}

# gated 仓库能不能真的下:元数据 200 不代表能下,要拿单个文件试 range GET。
can_download() {           # $1 repo  $2 仓库内路径 -> 打印 HTTP 码
    local repo="$1" remote="$2" tok
    tok="$(hf_token)"
    local url="https://huggingface.co/$repo/resolve/main/$remote"
    if [ -n "$tok" ]; then
        curl -sL -o /dev/null -w '%{http_code}' -r 0-1023 -H "Authorization: Bearer $tok" "$url"
    else
        curl -sL -o /dev/null -w '%{http_code}' -r 0-1023 "$url"
    fi
}

precheck() {
    local need_icl="$1"
    echo "==================== 前置体检 ===================="
    mountpoint -q "$EBS" || { echo "✗ $EBS 没挂载,先挂持久盘"; return 1; }
    echo "磁盘余量:"; df -h "$EBS" | tail -1

    local tok; tok="$(hf_token)"
    if [ -z "$tok" ]; then
        cat <<'EOF'
✗ 没有 HF token。二选一:
    hf auth login                  # 交互式贴 token(推荐)
    export HF_TOKEN=hf_xxxxx
EOF
        return 1
    fi
    echo "✓ 有 HF token ($("$HF" auth whoami 2>/dev/null | head -1))"

    local code
    code="$(can_download "$REPO_MAIN" vae/ltx-2.5-audio-vae-bf16.safetensors)"
    if [ "$code" != "200" ] && [ "$code" != "206" ]; then
        echo "✗ $REPO_MAIN 取文件返回 HTTP $code —— 还没接受条款。去点 Agree and Access:"
        echo "    https://huggingface.co/$REPO_MAIN"
        return 1
    fi
    echo "✓ $REPO_MAIN 可下载 (HTTP $code)"

    if [ "$need_icl" = yes ]; then
        code="$(can_download "$REPO_ICL" ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors)"
        if [ "$code" != "200" ] && [ "$code" != "206" ]; then
            echo "✗ $REPO_ICL 取文件返回 HTTP $code —— 还没接受条款:"
            echo "    https://huggingface.co/$REPO_ICL"
            return 1
        fi
        echo "✓ $REPO_ICL 可下载 (HTTP $code)"
    fi
}

# ---------------------------------------------------------------- 收尾体检
report() {
    echo "==================== 软链体检 ===================="
    local row repo remote kind name p
    for row in "${MANIFEST_BASE[@]}" "${MANIFEST_ICL[@]}" "${MANIFEST_ENH[@]}"; do
        IFS='|' read -r repo remote kind name <<<"$row"
        p="$COMFY/models/$kind/$name"
        if [ -f "$p" ];   then printf "  OK      %s/%s\n" "$kind" "$name"
        elif [ -L "$p" ]; then printf "  BROKEN  %s/%s\n" "$kind" "$name"
        else                   printf "  MISSING %s/%s\n" "$kind" "$name"; fi
    done
    cat <<'EOF'

工作流(API 格式,用 curl POST /prompt 跑):
  user/default/workflows/ltx_2.5/i2v.api.json                 图生视频(两阶段,带同步音频)
  user/default/workflows/ltx_2.5/multiref_ingredients.api.json 多参考(拼图 reference sheet + IC-LoRA)
说明:user/default/workflows/ltx_2.5/README.md

下完记得让 ComfyUI 重新扫一遍模型目录(重启或在 UI 里 Refresh),
否则 COMBO 里看不到新文件、/prompt 会报 value not in list。
EOF
}

TARGETS=("$@")
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=(base iclora)

# check 只体检
if [ "${TARGETS[0]}" = check ]; then
    precheck yes || true
    report
    exit 0
fi

need_icl=no
for t in "${TARGETS[@]}"; do [ "$t" = iclora ] && need_icl=yes; done
precheck "$need_icl"

for t in "${TARGETS[@]}"; do
    case "$t" in
        base)     echo "==================== LTX-2.5 基座 ===================="
                  run_manifest MANIFEST_BASE;;
        iclora)   echo "==================== Ingredients IC-LoRA ===================="
                  run_manifest MANIFEST_ICL;;
        enhancer) echo "==================== 提示词增强器(可选)===================="
                  run_manifest MANIFEST_ENH;;
        *) echo "未知目标: $t (可选 base / iclora / enhancer / check)"; exit 1;;
    esac
done

report

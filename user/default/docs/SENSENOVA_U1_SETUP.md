# SenseNova-U1 在 ComfyUI 中的环境记录(可复现 / Docker 用)

> 记录于 2026-06-01。目的:复现"ComfyUI + SenseNova-U1 本地图片编辑"环境,便于打 Docker 镜像。
> 宿主机 GPU:NVIDIA L40S(48GB),驱动 595.71.05。

---

## 0. 总览(一句话)

在一个 **Python 3.11 的 conda 环境 `sensenova`** 里,装好 ComfyUI 依赖 + SenseNova 官方 `sensenova-u1` 包(钉死 torch 2.8.0 / transformers 4.57.1),把官方 `apps/comfyui/` 作为自定义节点软链进 `custom_nodes/`,模型权重从 HuggingFace 拉 `sensenova/SenseNova-U1-8B-MoT`(约 33GB)。

**为什么不用默认环境**:模型的 `trust_remote_code` 自定义代码按 transformers **4.57.1** 写,在 transformers 5.x 上会崩(`NEOLLMConfig has no attribute 'rope_theta'`)。所以必须钉版。

---

## 1. 关键版本(已核实)

| 组件 | 版本 |
|---|---|
| Python | 3.11.15 |
| torch | 2.8.0+cu128 |
| torchvision | 0.23.0+cu128 |
| torchaudio | 2.8.0+cu128 |
| transformers | 4.57.1 |
| tokenizers | 0.22.1 |
| accelerate | 1.10.1 |
| safetensors | 0.6.2 |
| numpy | 2.4.6 |

> ⚠️ **torchaudio 必须显式对齐到 2.8.0+cu128**。`sensenova-u1` 的 pyproject 只钉了 torch/torchvision,torchaudio 是 ComfyUI requirements 带进来的,若残留旧版会触发 ABI 错误(`undefined symbol: torch_library_impl`),ComfyUI 启动即崩。

完整依赖清单见同目录 `sensenova_env_freeze.txt`(`pip freeze`,125 个包)。

---

## 2. 代码仓库版本(已核实)

| 仓库 | 路径 | Commit |
|---|---|---|
| ComfyUI | `/home/ubuntu/projects/mead/ComfyUI` | `b6b73c8b` (master) |
| SenseNova-U1 | `/home/ubuntu/SenseNova-U1` | `fd26e6db` (main, 2026-05-30) |

SenseNova-U1 用浅克隆(`git clone --depth 1 https://github.com/OpenSenseNova/SenseNova-U1`),官方 ComfyUI 集成在其 `apps/comfyui/` 子目录。

---

## 3. 下载的模型(已核实)

- **HuggingFace 模型**:`sensenova/SenseNova-U1-8B-MoT`
- **缓存位置**:`~/.cache/huggingface/hub/models--sensenova--SenseNova-U1-8B-MoT/`
- **快照 commit**:`bfa9b436503cb8aed4f2bc60e3236710cc77468d`
- **大小**:约 **33GB**(其中 8 个 safetensors 分片合计 33GB)
- **内容**:`model-0000{1..8}-of-00008.safetensors` + `config.json` / `tokenizer_config.json` / `vocab.json` / `merges.txt` / `added_tokens.json` / `special_tokens_map.json` / `chat_template.jinja` / `model.safetensors.index.json`

> Docker 注意:33GB 权重不打进镜像层。`Dockerfile.sensenova-u1` 配套的 `entrypoint.sh`
> 会在容器启动时检查 HF 缓存,缺失则自动下载;挂载缓存卷可让权重只下一次、跨容器复用。

### ⚠️ 容易混淆:SenseNova 是【两个组件】

打镜像时常见疑问"为什么节点路径不在 custom_nodes 下 / 推理包又在哪"——因为它由两部分组成,放法不同:

| 组件 | 是什么 | 放哪 | Dockerfile 对应 |
|---|---|---|---|
| 推理包 `sensenova_u1` | 模型推理代码(`it2i_generate` 等),pip 包,源码在 repo 的 `src/` | 装进 Python 环境(`pip install -e`) | 第 3b 步 |
| ComfyUI 节点 `apps/comfyui/` | 节点定义 `nodes.py`,ComfyUI 才认 | **必须**进 `custom_nodes/ComfyUI-SenseNova-U1` | 第 4 步 `install.py --copy` |

- `install.py --copy` 把节点**复制**进 `custom_nodes/`(容器内自包含,不是软链)。
- `--copy` 模式下节点目录里**没有** `src/`,推理包靠 pip 安装找到;因为用的是 `pip install -e`(editable),运行时仍依赖 `/opt/SenseNova-U1` 在原位,故该 repo **构建后不能删**,并通过 `SENSENOVA_U1_SRC` 显式指向其 `src/`。

---

## 4. SenseNova-U1 暴露的 ComfyUI 节点(我没有自己写方法,全部是官方提供)

> 重要澄清:**我没有手写任何推理方法/节点**。SenseNova 官方在 `apps/comfyui/` 里已提供完整集成,我只是安装、配置、修依赖、放工作流。底层推理走官方 `sensenova_u1` 包的 `it2i_generate`(图片编辑)/ `t2i_generate` / `interleave_gen`。

注册成功的 10 个节点:

| 节点类名 | 显示名 | 用途 |
|---|---|---|
| `SenseNovaU1LocalLoader` | SenseNova U1 Local Loader | 加载本地/HF 权重 |
| `SenseNovaU1LocalImageEdit` | SenseNova U1 Local Image Edit | **图片编辑**(`it2i_generate`) |
| `SenseNovaU1LocalTextToImage` | SenseNova U1 Local Text to Image | 文生图(`t2i_generate`) |
| `SenseNovaU1LocalInterleave` | SenseNova U1 Local Interleave | 图文交错生成 |
| `SenseNovaImageGenerate` | SenseNova Image Generate | 调 U1-Fast API |
| `SenseNovaChat` / `SenseNovaVisionURL` / `SenseNovaVisionImage` | — | API 工具节点 |
| `SenseNovaPromptBuilder` | SenseNova Prompt Builder | prompt 改写 |
| `SenseNovaInterleavePreview` | SenseNova Interleave Preview | 交错结果预览 |

节点入口:`apps/comfyui/nodes.py` 里的 `comfy_entrypoint()` → `SenseNovaExtension`(走 ComfyUI v3 节点 API `comfy_api.latest`)。

---

## 5. 我创建 / 放置的文件(已核实)

| 文件 | 说明 |
|---|---|
| `custom_nodes/ComfyUI-SenseNova-U1` → `/home/ubuntu/SenseNova-U1/apps/comfyui` | **符号链接**(由 `apps/comfyui/install.py` 创建)。⚠️ 移动/删除 repo 会断链,需重跑 install.py。Docker 里改用 `--copy`(见 `dockerfiles/Dockerfile.sensenova-u1`)。 |
| `user/default/workflows/sense-nova-u1/SenseNova_U1_image_edit.json` | 图片编辑工作流(复制自官方 `example_workflows/editing.json`) |
| `user/default/workflows/sense-nova-u1/{t2i,editing,interleave,infographic,api_u1_fast_t2i}.json` | 官方示例工作流(参考) |
| `input/1.webp` | 示例输入图(官方样例) |
| `output/sensenova_smoke_test.png` | 冒烟测试产物(可删) |
| `user/default/docs/sensenova_env_freeze.txt` | 完整 pip freeze(本次新生成) |
| `user/default/docs/SENSENOVA_U1_SETUP.md` | 本文档 |
| `user/default/dockerfiles/Dockerfile.sensenova-u1` | 可复现镜像构建文件 |

> 另:`test` conda 环境也装过节点,但 transformers 5.9 跑不通该模型,**Docker 复现请忽略 test 环境**。

---

## 6. 从零复现步骤(Docker / 全新机器)

> 现成的 Dockerfile 见 `user/default/dockerfiles/Dockerfile.sensenova-u1`(已固定 commit、做好依赖钉版与 torchaudio 对齐、含构建期自检)。下面是等价的手动步骤。

```bash
# --- 基础镜像建议 ---
# CUDA 12.8 runtime + cuDNN,Ubuntu 22.04/24.04,Python 3.11

# 1. 拉代码
git clone https://github.com/comfyanonymous/ComfyUI.git /opt/ComfyUI          # 或固定到 commit b6b73c8b
git clone --depth 1 https://github.com/OpenSenseNova/SenseNova-U1.git /opt/SenseNova-U1

# 2. 建环境(用 conda 或 venv 均可,这里给 conda 等价)
conda create -n sensenova python=3.11 -y
conda activate sensenova

# 3. 装 ComfyUI 依赖
pip install -r /opt/ComfyUI/requirements.txt

# 4. 装 sensenova-u1(钉版,会把 torch 降到 2.8.0 / transformers 到 4.57.1)
pip install httpx python-dotenv
pip install -e /opt/SenseNova-U1

# 5. 关键修复:对齐 torchaudio(否则 ComfyUI 启动崩)
pip install "torchaudio==2.8.0" --index-url https://download.pytorch.org/whl/cu128

# 6. 装 ComfyUI 自定义节点(Docker 推荐 --copy,避免软链依赖宿主路径)
python /opt/SenseNova-U1/apps/comfyui/install.py --comfyui /opt/ComfyUI --copy
#   --copy 模式下需设环境变量指向 src:
export SENSENOVA_U1_SRC=/opt/SenseNova-U1/src

# 7. 放工作流(整理在 sense-nova-u1 子目录下)
mkdir -p /opt/ComfyUI/user/default/workflows/sense-nova-u1
cp /opt/SenseNova-U1/apps/comfyui/example_workflows/*.json \
   /opt/ComfyUI/user/default/workflows/sense-nova-u1/

# 8. 模型权重(二选一)
#  a) 运行时挂载已有 HF 缓存卷到 ~/.cache/huggingface (推荐,33GB 不入镜像)
#  b) 构建/首启时预拉:
#     huggingface-cli download sensenova/SenseNova-U1-8B-MoT

# 9. 启动
python /opt/ComfyUI/main.py --listen 0.0.0.0
```

> 如果用纯 venv 不用 conda:把第 2 步换成 `python3.11 -m venv /opt/venv && . /opt/venv/bin/activate`,其余相同。

---

## 7. 复现后自检

```bash
# 依赖版本对不对
python -c "import torch,torchaudio,transformers; print(torch.__version__, torchaudio.__version__, transformers.__version__)"
# 期望: 2.8.0+cu128 2.8.0+cu128 4.57.1

# 模型自定义代码能否在 transformers 4.57.1 下导入
python -c "import sensenova_u1; from sensenova_u1.utils import load_model_and_tokenizer; print('ok')"

# 命令行端到端冒烟(快,num_steps=2)
cd /opt/SenseNova-U1 && python examples/editing/inference.py \
  --model_path sensenova/SenseNova-U1-8B-MoT \
  --prompt "Change the animal's fur color to a darker shade." \
  --image examples/editing/data/images/1.webp \
  --num_steps 2 --output /tmp/smoke.png
# 出图 = 通过
```

---

## 8. 已知坑(复现时会遇到)

1. **transformers 必须 4.57.1**,5.x 会 `rope_theta` AttributeError。
2. **torchaudio 必须 2.8.0**,跟 torch 主版本对齐,否则 ABI undefined symbol。
3. **节点要求 ComfyUI v3 API**(`comfy_api.latest` + `comfy_entrypoint`),太老的 ComfyUI 不加载这些节点。
4. **"Loading checkpoint shards 100%" 不是卡死**:之后是把权重搬上 GPU + 采样循环,SenseNova 自己的采样循环不打印逐步进度条,看 `nvidia-smi` 利用率确认在跑。正式出图 `num_steps=50`,L40S 上单张约 1 分钟级。
5. **conda TOS**:首次用官方 channel 需 `conda tos accept`(若用 venv 则无此问题)。

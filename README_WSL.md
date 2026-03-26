# IMAGGarment-1 WSL 安装与运行说明（重写版）

这份 README 基于项目原始 README 的核心信息重新整理，目标是让 **WSL2 + Ubuntu** 环境更稳定、更容易复现。

相较于原版，这一版重点解决了以下实际问题：

- `requirements.txt` 里同时包含 PyTorch / torchvision / torchaudio / CUDA 运行时固定版本，容易与本机 CUDA 或 conda 安装混装。
- `detectron2` 在安装时要求环境里已经有可用的 `torch`。
- `clip==1.0` 在 PyPI 上会和 OpenAI CLIP 混淆，推荐直接从 OpenAI CLIP 源码安装。
- `transformers`、`CLIP` 这类 GitHub 源码依赖在国内网络环境下容易因为 TLS / HTTP2 传输失败而中断。
- `deepspeed` 安装时会检查 `CUDA_HOME` / `nvcc`，如果只是为了先跑推理，不建议放在第一轮环境安装里。
- `pyairports==2.1.1` 当前不可直接从 PyPI 安装。

---

## 1. 原始 README 的关键信息

项目原 README 中明确给出的基础要求和命令主要有：

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA >= 11.8
- 用 conda 创建 `python=3.8.8` 环境
- 执行 `pip install -r requirements.txt`
- 下载主模型与额外组件模型
- 训练时运行 `train_color_adapter.sh`、`train_GAM.sh`、`train_LEM.sh`
- 测试时运行 `inference_IMAGGarment-1.py`

这份重写版保留这些核心流程，但把安装顺序和依赖拆分得更稳。

---

## 2. 推荐环境

- Windows 11 / Windows 10 + **WSL2**
- Ubuntu 22.04 或 20.04
- NVIDIA GPU（如需 GPU 推理/训练）
- **Windows 宿主机**安装 NVIDIA 驱动；WSL 内只使用 CUDA runtime / toolkit，不在 WSL 内额外安装 Linux 显卡驱动

先在 Windows PowerShell 中检查：

```powershell
wsl -l -v
nvidia-smi
```

在 Ubuntu/WSL 中再检查一次：

```bash
nvidia-smi
```

---

## 3. 安装系统依赖

进入 WSL Ubuntu 后执行：

```bash
sudo apt update
sudo apt install -y git wget curl unzip build-essential ninja-build
```

如果你还没有 Miniconda / Anaconda，请先安装 Miniconda。

---

## 4. 克隆项目

```bash
git clone https://github.com/muzishen/IMAGGarment-1.git
cd IMAGGarment-1
```

如果 GitHub 访问不稳定，建议提前设置：

```bash
git config --global http.version HTTP/1.1
```

---

## 5. 创建 conda 环境

```bash
conda create -n IMAGGarment python=3.8.8 -y
conda activate IMAGGarment
python -m pip install -U pip setuptools wheel
```

---

## 6. 先安装 PyTorch（单独安装，不要写进 requirements_wsl.txt）

**务必先装匹配版本的 PyTorch / torchvision / torchaudio。**

推荐使用一组明确匹配的版本：

### 方案 A：CUDA 12.1（推荐）

```bash
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

### 方案 B：CUDA 11.8

```bash
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

安装后验证：

```bash
python -c "import torch, torchvision; print(torch.__version__, torch.version.cuda); print(torchvision.__version__); print(torch.cuda.is_available())"
```

如果这里都不通过，不要继续后续步骤。

---

## 7. 安装主 requirements

本目录附带了一份新的 `requirements_wsl.txt`，它已经移除了在 WSL 初次安装中最容易出错的项目：

- torch / torchvision / torchaudio
- detectron2
- transformers（GitHub commit 安装）
- clip（OpenAI CLIP 源码安装）
- deepspeed
- vllm-flash-attn
- `nvidia-*-cu12` 这类显式 CUDA runtime pin
- pyairports

直接执行：

```bash
python -m pip install -r requirements_wsl.txt
```

---

## 8. 安装额外依赖（单独安装）

### 8.1 安装 OpenAI CLIP

推荐方式：

```bash
mkdir -p ~/src
cd ~/src
rm -rf CLIP
git clone https://github.com/openai/CLIP.git
cd CLIP
python -m pip install .
```

如果网络导致 `git clone` 失败，可以换成 zip 下载后本地安装：

```bash
cd ~/src
wget -O clip.zip https://github.com/openai/CLIP/archive/refs/heads/main.zip
python -m zipfile -e clip.zip .
cd CLIP-main
python -m pip install .
```

验证：

```bash
cd /mnt/d/fuxian/IMAGGarment-1
python -c "import clip; print('clip ok')"
```

---

### 8.2 安装 transformers（固定到项目使用的 commit）

项目原 requirements 使用的是：

- `78b2929c0554b79e0489b451ce4ece14d265ead2`

如果 `git clone` 大仓库经常失败，推荐直接下载指定 commit 的 zip：

```bash
cd ~/src
rm -rf transformers transformers-78b2929c0554b79e0489b451ce4ece14d265ead2 transformers.zip
wget -O transformers.zip https://github.com/huggingface/transformers/archive/78b2929c0554b79e0489b451ce4ece14d265ead2.zip
python -m zipfile -e transformers.zip .
cd transformers-78b2929c0554b79e0489b451ce4ece14d265ead2
python -m pip install .
```

验证：

```bash
cd /mnt/d/fuxian/IMAGGarment-1
python -c "import transformers; print(transformers.__version__)"
```

---

### 8.3 安装 detectron2

`detectron2` 需要在 **torch 已经可用** 的前提下再安装。推荐源码安装：

```bash
cd /mnt/d/fuxian/IMAGGarment-1
python -m pip install --no-build-isolation "git+https://github.com/facebookresearch/detectron2.git@ebe8b45437f86395352ab13402ba45b75b4d1ddb"
```

验证：

```bash
python -c "import detectron2; print('detectron2 ok')"
```

---

### 8.4 可选：安装 xformers

如果你后续需要 xformers，建议在 **PyTorch 已经装好且版本固定** 后再单独安装：

```bash
python -m pip install xformers==0.0.28.post1
```

如果你已经能正常运行推理，且没有性能瓶颈，也可以先不装。

---

### 8.5 可选：安装 deepspeed

`deepspeed` 安装时通常会检查 `CUDA_HOME` / `nvcc`。如果你当前目标只是先跑通推理，可以先不装。

如果后续训练需要 deepspeed，请先确认：

```bash
which nvcc
echo $CUDA_HOME
```

然后再单独安装：

```bash
python -m pip install deepspeed==0.15.4
```

---

## 9. 环境自检

全部装完后，建议一次性验证：

```bash
python -c "import torch, torchvision; print(torch.__version__, torch.version.cuda); print(torchvision.__version__)"
python -c "import clip; print('clip ok')"
python -c "import transformers; print(transformers.__version__)"
python -c "import detectron2; print('detectron2 ok')"
```

只要这 4 条都通过，说明环境主体已经完成。

---

## 10. 下载模型

根据原 README，需要准备以下模型：

- 主模型：百度云中的 IMAGGarment 权重
- `stabilityai/sd-vae-ft-mse`
- 训练：`stable-diffusion-v1-5/stable-diffusion-v1-5`
- 测试：`SG161222/Realistic_Vision_V4.0_noVAE`
- `h94/IP-Adapter`
- `stable-diffusion-v1-5/stable-diffusion-inpainting`

请把这些模型路径整理好，后续测试和训练都需要用到。

---

## 11. 如何测试

原 README 的测试入口如下：

```bash
python inference_IMAGGarment-1.py \
  --GAM_model_ckpt [GAM checkpoint] \
  --LEM_model_ckpt [LEM checkpoint] \
  --sketch_path [your sketch path] \
  --logo_path [your logo path] \
  --mask_path [your mask path] \
  --color_path [your color path] \
  --prompt [your prompt] \
  --output_path [your save path] \
  --color_ckpt [color adapter checkpoint] \
  --device [your device]
```

建议先用最小样例测试一遍，确保权重路径和输入路径都正确。

---

## 12. 如何训练

原 README 的训练流程如下：

```bash
# 请先下载 GarmentBench，并修改脚本中的路径

# train color adapter
sh train_color_adapter.sh
python change.py

# train GAM model
sh train_GAM.sh

# train LEM model
sh train_LEM.sh
```

开始训练前，请至少确认：

- GarmentBench 数据集已下载完成
- `train_color_adapter.sh` 中的数据路径已修改
- `train_GAM.sh` 中的路径已修改
- `train_LEM.sh` 中的路径已修改
- 权重输出目录具有写权限

---

## 13. 常见报错与处理

### 13.1 `ModuleNotFoundError: No module named 'torch'`（安装 detectron2 时）

先安装 PyTorch，再装 detectron2。

### 13.2 `clip==1.0` 无法从 PyPI 安装

不要直接用 PyPI 上的 `clip==1.0`。请从 OpenAI CLIP 源码安装。

### 13.3 `transformers` / `CLIP` GitHub clone 中断

可先执行：

```bash
git config --global http.version HTTP/1.1
```

仍不稳定时，改用指定 commit / 分支 zip 包方式安装。

### 13.4 `torch` 与 `torchvision` CUDA 版本不一致

必须重装成同一组版本，例如：

```bash
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

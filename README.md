# ECE1512 - Project A (SSM and VLM)
Yutian Mei yt.mei@mail.utoronto.ca
Jiachen Rao jc.rao@mail.utoronto.ca

## VMamba (SS2D) – Lightweight Experiments on ImageNet (val-only)

This part accompanies our course paper on extending **Mamba** to vision via **VMamba (SS2D)** and (theoretically) **Vim**. We focus on **accuracy/efficiency** under tight compute by training small VMamba variants on a **reduced ImageNet (ILSVRC2012) validation set**.

---

### What’s inside
- **Paper outline**: Motivation, related work, how Mamba is extended by VMamba (SS2D) and Vim (bi-scan + pos. enc.), plus proposed experimental setups.
- **Experiments (run)**: VMamba **4-dir SS2D** vs **2-dir** ablation on ImageNet-val-only (20 classes).  
- **Experiments (theory-only)**: Vim variants and ablations (no runs due to compute).

---

### Dataset protocol (small & reproducible)
- Source: **ImageNet (ILSVRC2012) validation** set only.  
- Split: **80% train / 20% val** (within val).  
- Class reduction: **1,000 → 20 classes** (subset) to cut compute.  
- Pretrained compatibility: we **strip the classifier head** via `strip_head.py` so `NUM_CLASSES` can differ from the checkpoint.

> Note on 4-dir vs 2-dir: In the official VMamba repo, `v05` (4-dir cross2d) depends on the `selective_scan_cuda_oflex` kernel. Without it, the code falls back to a non-oflex path that behaves like **2-dir**. On our setup (CUDA / selective-scan mismatch), `v05` ≈ `v052d` → **same accuracy, similar throughput**.

---

### Models
https://drive.google.com/drive/folders/1Btr66YbMwRaDLI21aIcszdBz0Hg2MoQE?usp=sharing
- **From scratch**: VMamba-Tiny **4-dir SS2D** and **2-dir** ablation.
- **Fine-tune**: Pretrained backbone **with stripped head**, resized inputs.

---

### Environment
- Python 3.10+, PyTorch ≥ 2.1.
- Clone upstream (or this fork):
  ```bash
  git clone https://github.com/MzeroMiko/VMamba
  cd VMamba
  conda create -n vmamba python=3.10
  pip install torch==2.2 torchvision torchaudio triton pytest chardet yacs termcolor fvcore seaborn packaging ninja einops numpy==1.24.4 timm==0.4.12
  pip install https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4+cu12torch2.2cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
  pip install https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.2.0.post2/causal_conv1d-1.2.0.post2+cu118torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
  pip install https://github.com/state-spaces/mamba/releases/download/v1.2.0.post1/mamba_ssm-1.2.0.post1+cu118torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
  python vmamba.py


### result summary

| Model                 | Dirs | ImgSize | Top-1 | Top-5 | img/s |
| --------------------- | :--: | :-----: | ----: | ----: | ----: |
| VMamba-Tiny (scratch) |   4  |   224  |  27.00 |  64.50 |  158.5 |
| VMamba-Tiny (scratch) |   2  |   224  |  27.00 |  64.50 |  160.3 |
| VMamba-Tiny (ft, pre) |   4  |   224  |  99.50 |  99.50 |  169.5 |
| VMamba-Tiny (ft, pre) |   4  |   128  |  92.50 |  97.50 |  338.3 |
| VMamba-Tiny (ft, pre) |   4  |    64  |  74.00 |  95.00 |  381.6 |



## LLaVA Projector Optimization: Efficiency and Expressiveness Study

This part of the repository contains the experimental code and analysis for a course project based on **[LLaVA: Visual Instruction Tuning (Liu et al., 2023)](https://arxiv.org/abs/2304.08485)**.  
We focus on identifying efficiency bottlenecks in LLaVA’s architecture and optimizing the **vision-to-language projection layer**, which bridges the CLIP vision encoder and the language model input space.

---

### Motivation

The original LLaVA model uses a **single linear projector** to map vision features (from CLIP ViT-L/14) into the LLM’s embedding space.  
While computationally simple, this linear mapping limits the model’s ability to capture complex cross-modal relationships.  
Our goal is to explore more expressive alternatives while analyzing their computational efficiency and representational quality.

---

### Experiment Overview

#### 1. **Profiling the Baseline Linear Projector**
We measure the runtime and FLOPs distribution of the CLIP vision encoder and linear projector to identify the main computational bottlenecks.

#### 2. **Parameter and Memory Comparison**
We define and compare several projector architectures:
- **Linear** – Original LLaVA mapping layer  
- **LowRank** – Low-rank factorization for parameter reduction  
- **MLP** – Two-layer MLP for nonlinear mapping  
- **Gated** – Sigmoid-gated linear fusion  
- **LoRA** – Low-rank adaptation for efficient fine-tuning  

Each architecture is evaluated for its parameter count and memory footprint.

#### 3. **Runtime Profiling**
Average forward-pass time per projector is measured on 200 random CLIP feature inputs to assess computational efficiency.

#### 4. **Reconstruction-Based Evaluation**
Using a **CIFAR-10 subset**, we train each projector to reconstruct CLIP vision embeddings, serving as a proxy for cross-modal alignment performance.

| Projector | Test Loss | Params (M) | FLOPs (M) | Runtime (s) |
|------------|--------------|-------------|-------------|--------------|
Linear     |     0.1719 |       1.05 |       1.05 |       7.90
LowRank    |     0.1414 |       0.52 |       0.52 |       7.99
MLP        |     0.1169 |       4.20 |       4.19 |      31.97
Gated      |     0.1773 |       2.10 |       2.10 |      13.27
LoRA       |     0.1874 |       1.07 |       1.57 |      12.74

---

### Key Insights

- The **linear projector** is not the primary computational bottleneck but limits expressiveness.  
- The **MLP projector** achieves the best reconstruction accuracy, supporting the use of nonlinear transformations for visual–language alignment.  
- The **LowRank** variant offers an excellent trade-off between parameter efficiency and accuracy.  
- **LLaVA-NeXT (2024)** adopted a **two-layer MLP projector**, consistent with our experimental findings.


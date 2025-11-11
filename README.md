# ECE1512 - Project A (SSM and VLM)
Yutian Mei yt.mei@mail.utoronto.ca
Jiachen Rao jc.rao@mail.utoronto.ca

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


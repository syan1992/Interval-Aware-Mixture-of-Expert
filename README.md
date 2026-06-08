# Interval-Aware Mixture of Experts (IAMoE)

A PyTorch implementation of an **Interval-Aware Mixture of Experts (MoE)** architecture inspired by DeepSeek-MoE, featuring dynamic/static ordered centroids for structured expert routing and label-interval-based gating supervision.

## Overview

This project implements a MoE module where expert routing is guided by **ordered centroids** in the embedding space, aligned with target value intervals. The key idea is to encourage each routed expert to specialize on a specific range of the target distribution, improving both interpretability and load balance.

### Key Components

| Module | Description |
|---|---|
| `models/centroids.py` | Static and dynamic ordered centroid providers for routing |
| `models/deepseek_moe.py` | Core MoE layer |
| `models/gating_loss.py` | Soft interval-based gating supervision loss (KL divergence) |
| `models/module.py` | High-level `IAMoE` wrapper module |
| `utils/util.py` | Dataset utilities: label normalization and quantile-based interval construction |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from utils.util import calmean
from models.module import IAMoE

# 1. Compute label statistics and quantile intervals from your dataset
mean, std, intervals, group_nums, group_labels, num_experts = calmean(dataset)

# 2. Build the model
model = IAMoE(
    input_feat=256,       # Input feature dimension
    dim_feat=128,         # Expert hidden dimension
    num_tasks=1,
    num_experts=4,        # Number of routed experts
    num_heads=1,
    output_feat=128,
    intervals=intervals,
)

# 3. Forward pass
gate, output = model(x, epoch=epoch)
```

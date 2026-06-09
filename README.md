# Interval-Aware Mixture of Experts (IA-MoE)

A PyTorch implementation of an **Interval-Aware Mixture of Experts (IA-MoE)** framework for regression tasks. IA-MoE introduces interval-aware routing supervision to encourage expert specialization across different regions of the target distribution, improving performance on imbalanced regression problems.

## Overview

IA-MoE extends conventional Mixture of Experts architectures by incorporating interval-based target decomposition and routing supervision. Target values are partitioned into ordered intervals, and experts are encouraged to specialize in specific regions of the target space through soft interval-aware gating targets.

## Project Structure

| Module | Description |
|----------|-------------|
| `models/centroids.py` | Static and dynamic ordered centroid construction for expert routing |
| `models/deepseek_moe.py` | Core MoE layer with Top-K expert routing |
| `models/gating_loss.py` | Interval-aware routing supervision loss |
| `models/module.py` | High-level `IAMoE` wrapper module |
| `utils/util.py` | Dataset preprocessing, label normalization, and interval construction |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from utils.util import calmean
from models.module import IAMoE
from models.gating_loss import CustomExpertMSELoss

import torch
import torch.nn.functional as F

# Compute label statistics and interval assignments
mean, std, intervals, group_nums, group_labels, num_experts = calmean(dataset)

# Build IA-MoE
model = IAMoE(
    input_feat=256,
    dim_feat=128,
    num_experts=num_experts,
)

# Interval-aware routing supervision loss
criterion = CustomExpertMSELoss(intervals=intervals)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):

    # Forward pass
    gate, output = model(x, epoch=epoch)

    # Task-specific prediction head
    predictions = predictor(output)

    # Regression loss
    loss_task = F.mse_loss(predictions, targets)

    # Interval-aware routing supervision
    # targets_denorm: denormalized regression labels
    loss_gate = criterion(targets_denorm, gate)

    # Joint optimization
    loss = loss_task + loss_gate

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```


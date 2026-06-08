import math
import numpy as np
from typing import Dict, Union, List, Set
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from sklearn.metrics import silhouette_score
from collections import Counter
from scipy.spatial.distance import cosine
from scipy.stats import norm

import torch
import torch.optim as optim
from torch import nn
from torch_geometric.data import Data
import pdb

def calmean(dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels = [dataset[i].y for i in range(len(dataset))]
    labels_tensor = torch.stack(labels).to(device)      
    mm = torch.mean(labels_tensor, dim=0)
    ss = torch.std(labels_tensor, dim=0)

    yy = ((labels_tensor - mm) / ss).squeeze().cpu().numpy() 
    num_samples = len(dataset)

    num_experts = 4
    idx_sorted = np.argsort(yy)
    yy_sorted = yy[idx_sorted]

    quantiles = np.linspace(0, 1, num_experts + 1)
    edges = np.quantile(yy_sorted, quantiles)

    intervals = []
    overlap=0
    for i in range(num_experts):
        left = edges[i]
        right = edges[i + 1]
        
        width = right - left
        extend = overlap * width
        
        new_left = left - extend
        new_right = right + extend
        
        intervals.append((float(new_left), float(new_right)))

    centers = [ (edges[i] + edges[i+1]) / 2.0 for i in range(len(edges)-1) ]

    bin_ids_sorted = np.digitize(yy_sorted, edges[1:-1], right=False)  # length N

    group_nums_orig = np.zeros(num_samples, dtype=int)
    group_labels_orig = np.zeros(num_samples, dtype=float)
    for pos_sorted, idx_orig in enumerate(idx_sorted):
        k = bin_ids_sorted[pos_sorted]
        group_nums_orig[idx_orig] = k
        group_labels_orig[idx_orig] = centers[k]

    group_nums = torch.tensor(group_nums_orig, dtype=torch.float32).unsqueeze(1).to(device)
    group_labels = torch.tensor(group_labels_orig, dtype=torch.float32).unsqueeze(1).to(device)

    return (
        mm.to(device),
        ss.to(device),
        intervals,
        group_nums,
        group_labels,
        num_experts,
    )

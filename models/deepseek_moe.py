import torch
import torch.nn as nn
import torch.nn.functional as F
from models.centroids import build_centroid_provider

# Note: This is a simplified version of communication balance loss
# For the complete implementation with proper token-device mapping
# the device-limited routing implementation
# and more efficient calculations, please contact the author
class Expert(nn.Module):
    """
    Position-wise Feed-Forward Networks
    This consists of two linear transformations with a ReLU activation in between.
    
    FFN(x) = max(0, xW1 + b1 )W2 + b2
    d_model: embedding dimension (e.g., 512)
    d_expert: expert dimension (e.g., 256)
    
    """
    def __init__(self, d_model, d_expert, d_out):
        super().__init__()
        self.d_model = d_model

        self.fc1 = nn.Linear(d_model, d_expert, bias=True)
        self.fc2 = nn.Linear(d_expert, d_out, bias=True)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, input):
        # check input and first FF layer dimension matching
        batch_size, seq_length, d_input = input.size()
        assert self.d_model == d_input, "d_model must be the same dimension as the input"
        # max(0, xW_1 + b_1)W_2 + b_2 
        return self.fc2(F.relu(self.fc1(input))) 

class MixtureOfExperts(nn.Module):
    """
    Mixture of Expert as in DeepSeek

    MoE(x) = x + \sum Expert^s_i(x) + \sum gate(x;K)*Expert^r_i(x)
    d_model: embedding dimension (e.g., 512)
    d_expert: expert dimension (e.g., 216)
    K : top K gate
    N_s: number of shared experts
    N_r: number of routed experts
    """
    def __init__(self, d_model, d_expert, d_out, K, N_s, N_r):
        super().__init__()

        self.d_model = d_model
        self.d_out = d_out
        self.K = K
        self.N_s = N_s
        self.N_r = N_r

        # initialize shared experts and routed experts
        self.shared_experts = nn.ModuleList([
            Expert(d_model, d_expert, d_out)
            for _ in range(N_s)
        ])

        self.routed_experts = nn.ModuleList([
            Expert(d_model, d_expert, d_out)
            for _ in range(N_r)
        ])

        self.centroid_engine = build_centroid_provider(
                mode="dynamic",   # "dynamic" / "static"
                N_r=N_r,
                d_model=d_model,
                scale_factor = 2.0,
                noise_level = 0
        )

    def forward(self, input, epoch = None):
        # check input and first FF layer dimension matching
        batch_size, seq_length, d_input = input.size()
        assert self.d_model == d_input, "d_model must be the same dimension as the input"

        
        shared_output = torch.zeros_like(input)[:, :, :self.d_out]
        for expert in self.shared_experts:
            shared_output += expert(input) 

        centroids = self.centroid_engine()
        self.similarities = torch.matmul(input, centroids.transpose(0, 1))  #[batch, seq, N_r]
        assert self.similarities.size(dim=-1) == self.N_r, \
        "last dimension of similarities must be the same as the number of routed expert"

        if epoch is None:
            T = 0.5
        else:
            T_init = 5.0
            T_min  = 0.5
            frac = min(1, epoch / 40)
            T = T_init * (1 - frac) + T_min * frac

        affinity = F.softmax(self.similarities / T, dim = -1)  

        ## Apply topK to calculate the gate 
        values, indexes = torch.topk(affinity, self.K)
        values = F.softmax(values, dim=-1) # Renormalize the top-K values
        gate = torch.zeros_like(affinity).scatter_(2, indexes, values)  #[batch, seq, N_r]
        """for testing"""

        self.last_gate = gate

        routed_output = torch.zeros_like(input)[:, :, :self.d_out]
        for i in range(self.N_r):
            routed_output += gate[:, :, i].unsqueeze(-1) * self.routed_experts[i](input)

        
        return routed_output, affinity

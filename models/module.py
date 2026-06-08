import torch
from models.deepseek_moe import MixtureOfExperts

class IAMoE(torch.nn.Module):
    def __init__(self, input_feat, dim_feat, num_experts):
        super(IAMoE, self).__init__()

        self.moe = MixtureOfExperts(
            d_model=input_feat,   # Input dimension
            d_expert=dim_feat,    # Expert hidden dimension
            d_out=dim_feat,       # Output dimension
            K=2,                  # Top-K experts per token
            N_s=0,                # Number of shared experts
            N_r=num_experts,      # Number of routed experts
        )

    def forward(self, x, epoch = None):
        x_input = x
        if len(x_input.shape) == 1:
            x_input = x_input.unsqueeze(0)
        
        f_moe, gate = self.moe(x_input.unsqueeze(1), epoch)
        return gate, f_moe.squeeze()

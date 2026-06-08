import torch

def build_soft_target_distribution_gating(y, intervals, sigma=0.2):
    """
    Soft target distribution where:
    - Distance inside interval = 0
    - Distance outside = distance to nearest boundary
    """
    device = y.device

    y = y.unsqueeze(1)  # [B,1]

    L = torch.tensor([l for (l, u) in intervals], device=device).unsqueeze(0)  # [1,E]
    U = torch.tensor([u for (l, u) in intervals], device=device).unsqueeze(0)  # [1,E]

    left_dist  = torch.clamp(L - y, min=0)  # y < l → positive distance
    right_dist = torch.clamp(y - U, min=0)  # y > u → positive distance

    dist = left_dist + right_dist
    soft = torch.exp(- dist**2 / (2 * sigma**2))
    soft = soft / (soft.sum(dim=1, keepdim=True) + 1e-8)

    return soft

class CustomExpertMSELoss(torch.nn.Module):
    def __init__(self, intervals=None):
        super().__init__()
        self.intervals = intervals

    def forward(self, targets, gate):
        sigma = 0.3
        loss_gating, w1 = gating_supervision_loss(
            gate.squeeze(),
            targets,
            self.intervals,
            loss_type="kl",
            sigma=sigma
        )
        return loss_gating
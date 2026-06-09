import torch
import torch.nn.functional as F

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

def gating_supervision_loss(gating_probs, targets_denorm, intervals, loss_type="kl", sigma=0.2):
    """
    gating_probs: [B, E] from softmax gating
    targets_denorm: [B] (denormalized y)
    intervals: list of (lower, upper)
    loss_type: "kl" or "mse"
    """
    soft_targets = build_soft_target_distribution_gating(targets_denorm, intervals, sigma=sigma)  # [B, E]
    if loss_type == "kl":
        loss = F.kl_div(torch.log(gating_probs + 1e-8), soft_targets, reduction='batchmean')

        cdf_p = soft_targets.cumsum(dim=1)
        cdf_q = gating_probs.cumsum(dim=1)
        w1 = (cdf_p - cdf_q).abs().sum(dim=1).mean()
    elif loss_type == "mse":
        loss = F.mse_loss(gating_probs, soft_targets)
        w1 = torch.tensor(0.0, device=gating_probs.device)
    else:
        raise ValueError("Unsupported loss_type: choose 'kl' or 'mse'")

    return loss
    
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

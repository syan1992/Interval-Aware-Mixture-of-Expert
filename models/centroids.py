import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseCentroidProvider(nn.Module):
    """
    Unified centroid interface.
    All MoE modules call provider() to get centroids.
    """

    def forward(self):
        raise NotImplementedError

class StaticOrderedCentroids(BaseCentroidProvider):

    def __init__(
        self,
        N_r,
        d_model,
        sample_features=None,
        scale_factor=1.0,
        sign_strength=0.1,
        normalize=False,
    ):
        super().__init__()

        # ---- base_mu ----
        if sample_features is not None:
            base_mu = sample_features.mean(
                dim=list(range(sample_features.dim() - 1))
            )
        else:
            std = 1.0 / math.sqrt(d_model)
            base_mu = torch.randn(d_model) * (std * 0.1)

        if normalize:
            base_mu = F.normalize(base_mu, dim=-1)

        norm_centers = torch.linspace(-1.0, 1.0, N_r, device=base_mu.device)

        direction = torch.randn(d_model, device=base_mu.device)
        bias = sign_strength * torch.sign(base_mu)
        direction = direction + bias

        dot_val = torch.dot(direction, base_mu)
        projection = (dot_val * base_mu / (torch.norm(base_mu) ** 2 + 1e-9))

        direction = direction - projection
        direction = F.normalize(direction, dim=-1)

        mu_scale = 1.0
        centroids = []
        for nc in norm_centers:
            c = base_mu + (nc * scale_factor * mu_scale) * direction
            centroids.append(c)

        centroids = torch.stack(centroids)
        centroids = centroids - centroids.mean(dim=0)
        centroids = F.normalize(centroids, dim=-1)
        self.centroids = nn.Parameter(centroids)

    def forward(self):
        return self.centroids

class DynamicOrderedCentroids(BaseCentroidProvider):

    def __init__(
        self,
        N_r,
        d_model,
        sample_features=None,
        scale_factor=1.0,
        sign_strength=0.1,
        noise_level=0.01,
        normalize_base_mu=True,
        center_centroids = True
    ):
        super().__init__()

        self.center_centroids = center_centroids
        self.N_r = N_r
        self.scale_factor = scale_factor
        self.noise_level = noise_level
        self.normalize_base_mu = normalize_base_mu
        self.eps = 1e-9

        # ---- base_mu ----
        if sample_features is not None:
            with torch.no_grad():
                init_base = sample_features.mean(
                    dim=list(range(sample_features.dim() - 1))
                )
        else:
            std = 1.0 / math.sqrt(d_model)
            init_base = torch.randn(d_model) * (std * 0.1)

        self.base_mu = nn.Parameter(init_base)

        init_dir = torch.randn(d_model)
        init_dir = init_dir + sign_strength * torch.sign(init_base)
        self.direction = nn.Parameter(init_dir)

        # ---- ordered gaps ----
        if N_r > 1:
            init_gap_val = self.scale_factor / (N_r - 1)
            init_param = math.log(math.exp(init_gap_val) - 1 + 1e-10)
            self.raw_gaps = nn.Parameter(torch.full((N_r - 1,), init_param))
        else:
            self.register_buffer("raw_gaps", torch.tensor([]))

    # ----------------------------------

    def get_centroids(self):

        b = self.base_mu
        if self.normalize_base_mu:
            b = F.normalize(b, dim=-1)

        d = self.direction
        dot_val = torch.dot(d, b)
        b_norm_sq = torch.norm(b) ** 2 + self.eps

        ortho_vec = d - (dot_val / b_norm_sq) * b
        if torch.norm(ortho_vec) < 1e-6:
            ortho_vec = ortho_vec + torch.ones_like(ortho_vec) * 1e-4
        direction_ortho = F.normalize(ortho_vec, dim=-1, eps=self.eps)

        # ordered offsets
        if self.N_r > 1:
            gaps = F.softplus(self.raw_gaps)
            t_raw = torch.cat([torch.zeros(1, device=b.device), torch.cumsum(gaps, dim=0)])
            total_gap_sum = gaps.sum() 
            t_adaptive = (t_raw - total_gap_sum / 2.0) * self.scale_factor
            if not self.normalize_base_mu:
                t_adaptive *= torch.norm(b)
        else:
            t_adaptive = torch.zeros(1, device=b.device)

        centroids = (b.unsqueeze(0) + t_adaptive.unsqueeze(1) * direction_ortho.unsqueeze(0))

        # training noise
        if self.training and self.noise_level > 0:
            avg_gap = self.scale_factor / max(self.N_r, 1)
            adaptive_noise = self.noise_level * avg_gap
            centroids = centroids + torch.randn_like(centroids) * adaptive_noise

        #if self.center_centroids:
        #    centroids = centroids - centroids.mean(dim=0, keepdim=True)
        centroids = F.normalize(centroids, dim=-1)
        return centroids

    def forward(self):
        return self.get_centroids()

def build_centroid_provider(
    mode="dynamic",
    **kwargs,
):
    """
    mode:
        - dynamic
        - static
    """

    if mode == "dynamic":
        return DynamicOrderedCentroids(**kwargs)

    elif mode == "static":
        return StaticOrderedCentroids(**kwargs)

    else:
        raise ValueError(f"Unknown centroid mode: {mode}")

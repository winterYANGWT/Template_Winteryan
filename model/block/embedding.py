import torch
import torch.nn as nn
import einops

__all__ = ['SinusoidalEmbedding', 'GuassianFourierEmbedding']


class SinusoidalEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        half_dim = embedding_dim // 2
        nlog_10000 = -torch.log(torch.tensor(10000.))
        w = torch.exp(torch.arange(half_dim) / half_dim * nlog_10000)
        self.register_buffer('w', w)

    def forward(self, t):
        wt = torch.einsum('N,D->ND', t, self.w)
        sin_wt, cos_wt = wt.sin(), wt.cos()
        embedding = einops.rearrange([sin_wt, cos_wt], 'L N D->N (L D)')
        return embedding


class GuassianFourierEmbedding(nn.Module):
    def __init__(self, embedding_dim, scale=30) -> None:
        super().__init__()
        half_dim = embedding_dim // 2
        w = torch.randn(half_dim) * scale
        self.register_buffer('w', w)

    def __init__(self, t):
        wt = torch.einsum('N,D->ND', t, self.w) * 2 * torch.pi
        sin_wt, cos_wt = wt.sin(), wt.cos()
        embedding = einops.rearrange([sin_wt, cos_wt], 'L N D->N (L D)')
        return embedding

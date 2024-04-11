'''reference: https://github.com/lucidrains/vit-pytorch/'''

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class PostNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.norm(self.fn(x, **kwargs))

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim=None, dropout = 0.):
        super().__init__()
        if out_dim is None:
            out_dim = dim

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., visualize=False):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.visualize = visualize

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        register_attn = attn.clone()
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        if self.visualize:
            return self.to_out(out), register_attn
        else:
            return self.to_out(out)


def apply_complex(fr, fi, input, dtype = torch.complex64):
    return (fr(input.real)-fi(input.imag)).type(dtype) \
            + 1j*(fr(input.imag)+fi(input.real)).type(dtype)

class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.fc_r = nn.Linear(in_features, out_features)
        self.fc_i = nn.Linear(in_features, out_features)

    def forward(self, input):
        return apply_complex(self.fc_r, self.fc_i, input)


class FreqFilter_multihead(nn.Module):
    '''freq filter layer w multihead design, either maxpooling or projection'''
    def __init__(self, head=4, length=64, mode='pool'):
        super().__init__()
        assert mode in ['pool', 'self']
        self.mode = mode
        self.head, self.dim = head, length
        self.complex_weight = nn.Parameter(torch.randn(head, length, 2, dtype=torch.float32) * 0.02)

        if self.mode == 'self':
            self.filter_weight = nn.Linear(length, head)

    def forward(self, x):
        '''input size: B x N x C'''
        B, N, C = x.shape
        device = x.device

        x = x.to(torch.float32)
        x = torch.fft.rfft(x, dim=(1), norm='ortho')

        if self.mode == 'self':
            head_weight = self.filter_weight(x.real)

        x = repeat(x, 'b p d -> b p h d', h=self.head)
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight

        if self.mode == 'pool':
            x = rearrange(x, 'b p h d -> (b p d) h')
            _, indices = F.max_pool1d(x.abs(), kernel_size=x.shape[-1], return_indices=True)
            batch_range = torch.arange(x.shape[0], device = x.device)[:, None]
            x = x[batch_range, indices][:, 0]
            x = rearrange(x, '(b p d) -> b p d', b=B, d=self.dim)

        elif self.mode == 'self':
            x = x * head_weight.unsqueeze(-1)
            x = torch.sum(x, dim=-2)

        x = torch.fft.irfft(x, n=N, dim=(1), norm='ortho')
        return x


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            out = attn(x)
            x = out + x
            x = ff(x) + x
        return x


class FA_Transformer(nn.Module):
    def __init__(
        self, 
        dim, 
        depth, 
        mlp_dim, 
        head=4, 
        mode='pool', 
        dropout = 0., 
        post_norm=False):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            if not post_norm:
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, FreqFilter_multihead(head=head, length=dim, mode=mode)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
                ]))
            else:
                self.layers.append(nn.ModuleList([
                    PostNorm(dim, FreqFilter_multihead(head=head, length=dim, mode=mode)),
                    PostNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
                ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

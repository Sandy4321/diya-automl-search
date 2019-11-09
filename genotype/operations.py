import torch.nn as nn

__all__ = ['none', 'identity']
__all__ += ['sep_conv', 'dil_conv', 'max_pool', 'avg_pool']
__all__ += ['attention']


class Zero(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.mul(0.)


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):
        assert len(x.size()) == 3
        x = x.transpose(0, 1)
        out, _ = self.attn(x, x, x)
        return out.transpose(0, 1)


def identity(size, ks):
    return Identity()


def none(size, ks):
    return Zero()


def sep_conv(size, ks):
    C = size[0]
    D = len(size) - 1
    if D == 1:
        return nn.Sequential(
            nn.Conv1d(C, C, ks, padding=ks//2, groups=C),
            nn.Conv1d(C, C, 1, padding=0)
        )
    elif D == 2:
        return nn.Sequential(
            nn.Conv2d(C, C, ks, padding=ks//2, groups=C),
            nn.Conv2d(C, C, 1, padding=0)
        )
    else:
        raise NotImplementedError


def dil_conv(size, ks):
    C = size[0]
    D = len(size) - 1
    if D == 1:
        return nn.Sequential(
            nn.Conv1d(C, C, ks, padding=ks - 1, dilation=2, groups=C),
            nn.Conv1d(C, C, 1, padding=0)
        )
    elif D == 2:
        return nn.Sequential(
            nn.Conv2d(C, C, ks, padding=ks - 1, dilation=2, groups=C),
            nn.Conv2d(C, C, 1, padding=0)
        )
    else:
        raise NotImplementedError


def max_pool(size, ks):
    D = len(size) - 1
    if D == 1:
        return nn.MaxPool1d(ks, 1, padding=ks//2)
    elif D == 2:
        return nn.MaxPool2d(ks, 1, padding=ks//2)
    else:
        raise NotImplementedError


def avg_pool(size, ks):
    D = len(size) - 1
    if D == 1:
        return nn.AvgPool1d(ks, 1, padding=ks//2, count_include_pad=False)
    elif D == 2:
        return nn.AvgPool2d(ks, 1, padding=ks//2, count_include_pad=False)
    else:
        raise NotImplementedError


def attention(size, ks):
    C = size[-1]
    D = len(size) - 1
    if D == 1:
        return MultiheadAttention(C, 8)
    else:
        raise NotImplementedError

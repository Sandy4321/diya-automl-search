from functools import partial
import torch
import torch.nn as nn

__all__ = ['none', 'identity']
__all__ += ['sep_conv_1', 'sep_conv_3', 'sep_conv_5']
__all__ += ['dil_conv_1', 'dil_conv_3', 'dil_conv_5']
__all__ += ['max_pool_3', 'max_pool_5']
__all__ += ['avg_pool_3', 'avg_pool_5']
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


class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def extra_repr(self):
        return 'p={}, inplace'.format(self.p)

    def forward(self, x):
        if self.train and self.p > 0:
            mask = torch.zeros_like(x).bernoulli_(1 - self.p)
            x.div_(1 - self.p).mul_(mask)
        return x


def identity(size):
    return Identity()


def none(size):
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


sep_conv_1 = partial(sep_conv, ks=1)
sep_conv_3 = partial(sep_conv, ks=3)
sep_conv_5 = partial(sep_conv, ks=5)


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


dil_conv_1 = partial(dil_conv, ks=1)
dil_conv_3 = partial(dil_conv, ks=3)
dil_conv_5 = partial(dil_conv, ks=5)


def max_pool(size, ks):
    D = len(size) - 1
    if D == 1:
        return nn.MaxPool1d(ks, 1, padding=ks//2)
    elif D == 2:
        return nn.MaxPool2d(ks, 1, padding=ks//2)
    else:
        raise NotImplementedError


max_pool_3 = partial(max_pool, ks=3)
max_pool_5 = partial(max_pool, ks=5)


def avg_pool(size, ks):
    D = len(size) - 1
    if D == 1:
        return nn.AvgPool1d(ks, 1, padding=ks//2, count_include_pad=False)
    elif D == 2:
        return nn.AvgPool2d(ks, 1, padding=ks//2, count_include_pad=False)
    else:
        raise NotImplementedError


avg_pool_3 = partial(avg_pool, ks=3)
avg_pool_5 = partial(avg_pool, ks=5)


def attention(size):
    C = size[-1]
    D = len(size) - 1
    if D == 1:
        return MultiheadAttention(C, 8)
    else:
        raise NotImplementedError

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


def identity(D, C, ks):
    return Identity()


def none(D, C, ks):
    return Zero()


def sep_conv(D, C, ks):
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


def dil_conv(D, C, ks):
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


def max_pool(D, C, ks):
    if D == 1:
        return nn.MaxPool1d(ks, padding=ks//2)
    elif D == 2:
        return nn.MaxPool2d(ks, padding=ks//2)
    else:
        raise NotImplementedError


def avg_pool(D, C, ks):
    if D == 1:
        return nn.AvgPool1d(ks, padding=ks//2, count_include_pad=False)
    elif D == 2:
        return nn.AvgPool2d(ks, padding=ks//2, count_include_pad=False)
    else:
        raise NotImplementedError


def attention(D, C, ks):
    if D == 1:
        return nn.MultiHeadAttention(C, 8)
    else:
        raise NotImplementedError

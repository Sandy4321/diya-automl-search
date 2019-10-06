import torch
import torch.nn as nn

OPS_1D = {
    'none': lambda C: Zero(1),
    'identity': lambda C: Identity(),
    'tanh': lambda C: DenseTanh(C),
    'relu': lambda C: DenseReLU(C),
    'sigmoid': lambda C: DenseSigmoid(C),
}

OPS_2D = {
    'none': lambda C, stride: Zero(stride),
    'skip_connect': lambda C, stride: Identity() if stride == 1 else FactorizedReduce(C),
    'sep_conv_3x3': lambda C, stride: SepConv(C, C, 3, stride, 1),
    'sep_conv_5x5': lambda C, stride: SepConv(C, C, 5, stride, 2),
    'sep_conv_7x7': lambda C, stride: SepConv(C, C, 7, stride, 3),
    'dil_conv_3x3': lambda C, stride: DilConv(C, C, 3, stride, 2, 2),
    'dil_conv_5x5': lambda C, stride: DilConv(C, C, 5, stride, 4, 2),
    'avg_pool_3x3': lambda C, stride: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3': lambda C, stride: nn.MaxPool2d(3, stride=stride, padding=1),
}


class Zero(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride != 1:
            x = x[:, :, ::self.stride, ::self.stride]
        return x.mul(0.)


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class DenseTanh(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.linear = nn.Linear(C, 2*C)

    def forward(self, x):
        return torch.tanh(self.linear(x))


class DenseReLU(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.linear = nn.Linear(C, 2*C)

    def forward(self, x):
        return torch.relu(self.linear(x))


class DenseSigmoid(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.linear = nn.Linear(C, 2*C)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


class FactorizedReduce(nn.Module):
    def __init__(self, C):
        super().__init__()
        assert C % 2 == 0
        self.conv_1 = nn.Conv2d(C, C // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C, C // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C)

    def forward(self, x):
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = torch.relu(self.bn(out))
        return out


class DilConv(nn.Module):
    def __init__(self, C, kernel_size, stride, padding, dilation):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C, bias=False),
            nn.Conv2d(C, C, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):
    def __init__(self, C, kernel_size, stride, padding):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=kernel_size, stride=stride, padding=padding, groups=C, bias=False),
            nn.Conv2d(C, C, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=False),
            nn.Conv2d(C, C, kernel_size=kernel_size, stride=1, padding=padding, groups=C, bias=False),
            nn.Conv2d(C, C, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.op(x)

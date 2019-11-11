from copy import deepcopy
import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, stem, cells, classifier):
        super().__init__()
        self.stem = deepcopy(stem)
        self.cells = deepcopy(cells)
        self.classifier = deepcopy(classifier)

    def forward(self, x):
        out = self.stem(x)
        for cell in self.cells:
            out = cell(out)
        out = out.view(out.size(0), -1)
        return self.classifier(out)


class Recurrent(nn.Module):
    def __init__(self, stem, cells, classifier):
        super().__init__()
        self.stem = deepcopy(stem)
        self.cells = deepcopy(cells)
        self.classifier = deepcopy(classifier)

    def forward(self, x):
        xs = torch.transpose(self.stem(x), 0, 1)
        xs = xs.view(*xs.shape, 1)
        h = torch.zeros_like(xs[0])
        for x in xs:
            for cell in self.cells:
                h = cell(x, h)
        out = h.view(h.size(0), -1)
        return self.classifier(out)

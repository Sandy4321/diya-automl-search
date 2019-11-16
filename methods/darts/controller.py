import numpy as np
import torch
import torch.nn as nn
from methods.darts.cell import Cell


class CNNController(nn.Module):
    def __init__(self, size, num_classes, args):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(size[0], args.dim, 3, padding=1),
            nn.Dropout2d(args.dropout)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(args.dropout),
            nn.Linear(
                args.dim*np.prod(size[1:]),
                num_classes
            )
        )
        size = tuple([args.dim, *size[1:]])
        cell = Cell('cnn', size, args.nodes)
        self.cells = nn.ModuleList([cell]*args.cells)
        self.type = cell.type
        self.args = args
        self.init_alphas()

    def init_alphas(self):
        n_agg = len(self.type.AGGREGATIONS.keys())
        self.alpha_agg = nn.Parameter(
            1e-3*torch.randn(self.args.nodes, n_agg)
        )

        n_ops = len(self.type.OPERATIONS.keys())
        n_ops *= len(self.type.ACTIVATIONS.keys())
        self.alpha_ops = nn.ParameterList()
        for i in range(self.args.nodes):
            self.alpha_ops.append(
                nn.Parameter(1e-3*torch.randn(i + 2, n_ops))
            )

        modules = nn.ModuleList([self.stem, self.cells, self.classifier])
        self.weights = nn.ParameterList(modules.parameters())
        self.alphas = nn.ParameterList([self.alpha_agg, *self.alpha_ops])

    def forward(self, x):
        s0 = s1 = self.stem(x)
        for cell in self.cells:
            w_agg = [torch.softmax(a, dim=-1) for a in self.alpha_agg]
            w_ops = [torch.softmax(a, dim=-1) for a in self.alpha_ops]
            s0 = s1 = cell(s0, s1, w_agg, w_ops)
        out = s1.view(s1.size(0), -1)
        return self.classifier(out)


class RNNController(CNNController):
    def __init__(self, size, num_classes, args):
        super().__init__(size, num_classes, args)
        self.stem = nn.Sequential(
            nn.Linear(size[1], args.dim),
            nn.Dropout(args.dropout)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(args.dropout),
            nn.Linear(
                args.dim,
                num_classes
            )
        )
        size = tuple([args.dim, 1])
        self.cells = Cell('rnn', size, args.nodes)
        self.type = self.cells.type
        self.init_alphas()

    def forward(self, x):
        xs = torch.transpose(self.stem(x), 0, 1)
        xs = xs.view(*xs.shape, 1)
        h = torch.zeros_like(xs[0])
        w_agg = [torch.softmax(a, dim=-1) for a in self.alpha_agg]
        w_ops = [torch.softmax(a, dim=-1) for a in self.alpha_ops]
        for x in xs:
            h = self.cells(x, h, w_agg, w_ops)
        out = h.view(h.size(0), -1)
        return self.classifier(out)


class TransformerController(CNNController):
    def __init__(self, size, num_classes, args):
        super().__init__(size, num_classes, args)
        self.stem = nn.Sequential(
            nn.Linear(size[1], args.dim),
            nn.Dropout(args.dropout)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(args.dropout),
            nn.Linear(
                args.dim*size[0],
                num_classes
            )
        )
        size = tuple([size[0], args.dim])
        cell = Cell('transformer', size, args.nodes)
        self.cells = nn.ModuleList([cell]*args.cells)
        self.type = cell.type
        self.args = args
        self.init_alphas()

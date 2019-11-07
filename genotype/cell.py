import torch
import torch.nn as nn
import genomes.operations as ops

IDX_SEP = 'i'
AGGREGATIONS = {
    '0': 'sum',
    '1': 'product'
}
OPERATIONS = {
    '0': 'sep_conv',
    '1': 'dil_conv',
    '2': 'max_pool',
    '3': 'avg_pool',
    '4': 'attention',
    '5': 'identity',
    '6': 'none'
}
KERNEL_SIZE = {
    '0': 1,
    '1': 3,
    '2': 5
}
ACTIVATIONS = {
    '0': nn.ReLU(),
    '1': nn.Sigmoid(),
    '2': nn.Tanh(),
    '3': ops.Identity()
}


class Node(nn.Module):
    def __init__(self, size, agg, op1, op2):
        super().__init__()
        self.agg = AGGREGATIONS[agg]
        self.norm1 = nn.LayerNorm(size)
        ks = KERNEL_SIZE[op1[1]]
        self.op1 = nn.Sequential(
            getattr(ops, OPERATIONS[op1[0]])(size, ks),
            ACTIVATIONS[op1[2]]
        )
        self.norm2 = nn.LayerNorm(size)
        ks = KERNEL_SIZE[op2[1]]
        self.op2 = nn.Sequential(
            getattr(ops, OPERATIONS[op2[0]])(size, ks),
            ACTIVATIONS[op2[2]]
        )

    def forward(self, x1, x2):
        out1 = self.op1(self.norm1(x1))
        out2 = self.op2(self.norm2(x2))
        if self.agg == 'sum':
            return out1 + out2
        elif self.agg == 'product':
            return out1*out2
        else:
            raise NotImplementedError


class CNNCell(nn.Module):
    D = 2
    AGGREGATIONS = {
        '0': 'sum',
    }
    OPERATIONS = {
        '0': 'sep_conv',
        '1': 'dil_conv',
        '2': 'max_pool',
        '3': 'avg_pool',
        '5': 'identity',
        '6': 'none'
    }
    KERNEL_SIZE = {
        '0': 1,
        '1': 3,
        '2': 5
    }
    ACTIVATIONS = {
        '0': nn.ReLU(),
        '3': ops.Identity()
    }

    def __init__(self, size, genome):
        super().__init__()
        assert len(size) - 1 == self.D
        self.idx = []
        self.nodes = nn.ModuleList([])
        for seq in genome:
            agg, seq = seq[0], seq[1:]
            idx1, seq = seq.split(IDX_SEP, 1)
            op1, seq = seq[:3], seq[3:]
            idx2, op2 = seq.split(IDX_SEP, 1)
            self.check_ops(agg, op1, op2)

            self.idx.append((int(idx1), int(idx2)))
            self.nodes.append(Node(size, agg, op1, op2))

    def forward(self, x):
        outs = [x]
        for (idx1, idx2), node in zip(self.idx, self.nodes):
            out = node(outs[idx1], outs[idx2])
            outs.append(out)
        return out

    def check_ops(self, agg, *ops):
        assert agg in self.AGGREGATIONS.keys()
        for op in ops:
            assert op[0] in self.OPERATIONS.keys()
            assert op[1] in self.KERNEL_SIZE.keys()
            assert op[2] in self.ACTIVATIONS.keys()


class RNNCell(CNNCell):
    D = 1
    AGGREGATIONS = {
        '0': 'sum',
        '1': 'product'
    }
    OPERATIONS = {
        '0': 'sep_conv',
        '5': 'identity',
        '6': 'none'
    }
    KERNEL_SIZE = {
        '0': 1,
    }
    ACTIVATIONS = {
        '0': nn.ReLU(),
        '1': nn.Sigmoid(),
        '2': nn.Tanh(),
        '3': ops.Identity()
    }

    def forward(self, x, h):
        x = torch.cat([x, h], dim=-1)
        outs = [x]
        for (idx1, idx2), node in zip(self.idx, self.nodes):
            out = node(outs[idx1], outs[idx2])
            outs.append(out)
        out, h = torch.chunk(out, 2, dim=-1)
        return out


class TransformerCell(CNNCell):
    D = 1
    AGGREGATIONS = {
        '0': 'sum',
    }
    OPERATIONS = {
        '0': 'sep_conv',
        '1': 'dil_conv',
        '4': 'attention',
        '5': 'identity',
        '6': 'none'
    }
    KERNEL_SIZE = {
        '0': 1,
        '1': 3,
        '2': 5
    }
    ACTIVATIONS = {
        '0': nn.ReLU(),
        '1': nn.Sigmoid(),
        '2': nn.Tanh(),
        '3': ops.Identity()
    }

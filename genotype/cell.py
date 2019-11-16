import torch.nn as nn
import genotype.operations as ops

IDX_SEP = 'i'
AGGREGATIONS = {
    '0': 'sum',
    '1': 'product'
}
OPERATIONS = {
    '00': 'none',
    '01': 'identity',
    '02': 'sep_conv_1',
    '03': 'sep_conv_3',
    '04': 'sep_conv_5',
    '05': 'dil_conv_1',
    '06': 'dil_conv_3',
    '07': 'dil_conv_5',
    '08': 'max_pool_3',
    '09': 'max_pool_5',
    '10': 'avg_pool_3',
    '11': 'avg_pool_5',
    '12': 'attention',
}
ACTIVATIONS = {
    '0': ops.Identity(),
    '1': nn.ReLU(),
    '2': nn.Sigmoid(),
    '3': nn.Tanh(),
}


class Node(nn.Module):
    def __init__(self, size, agg, op1, op2):
        super().__init__()
        self.agg = AGGREGATIONS[agg]
        self.norm1 = nn.LayerNorm(size)
        self.op1 = nn.Sequential(
            getattr(ops, OPERATIONS[op1[:2]])(size),
            ACTIVATIONS[op1[2]]
        )
        self.norm2 = nn.LayerNorm(size)
        self.op2 = nn.Sequential(
            getattr(ops, OPERATIONS[op2[:2]])(size),
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
    D = 3
    AGGREGATIONS = {
        '0': 'sum',
    }
    OPERATIONS = {
        '00': 'none',
        '01': 'identity',
        '02': 'sep_conv_1',
        '03': 'sep_conv_3',
        '04': 'sep_conv_5',
        '05': 'dil_conv_1',
        '06': 'dil_conv_3',
        '07': 'dil_conv_5',
        '08': 'max_pool_3',
        '09': 'max_pool_5',
        '10': 'avg_pool_3',
        '11': 'avg_pool_5',
    }
    ACTIVATIONS = {
        '0': ops.Identity(),
        '1': nn.ReLU(),
    }

    def __init__(self, size, genome):
        super().__init__()
        assert len(size) == self.D
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
        return outs[-1]

    def check_ops(self, agg, *ops):
        assert agg in self.AGGREGATIONS.keys()
        for op in ops:
            assert op[:2] in self.OPERATIONS.keys()
            assert op[2] in self.ACTIVATIONS.keys()


class RNNCell(CNNCell):
    D = 2
    AGGREGATIONS = {
        '0': 'sum',
        '1': 'product'
    }
    OPERATIONS = {
        '00': 'none',
        '01': 'identity',
        '02': 'sep_conv_1',
    }
    ACTIVATIONS = {
        '0': ops.Identity(),
        '1': nn.ReLU(),
        '2': nn.Sigmoid(),
        '3': nn.Tanh(),
    }

    def forward(self, x, h):
        outs = [x, h]
        for (idx1, idx2), node in zip(self.idx, self.nodes):
            out = node(outs[idx1], outs[idx2])
            outs.append(out)
        return outs[-1]


class TransformerCell(CNNCell):
    D = 2
    AGGREGATIONS = {
        '0': 'sum',
    }
    OPERATIONS = {
        '00': 'none',
        '01': 'identity',
        '02': 'sep_conv_1',
        '03': 'sep_conv_3',
        '04': 'sep_conv_5',
        '05': 'dil_conv_1',
        '06': 'dil_conv_3',
        '07': 'dil_conv_5',
        '08': 'max_pool_3',
        '09': 'max_pool_5',
        '10': 'avg_pool_3',
        '11': 'avg_pool_5',
        '12': 'attention',
    }
    ACTIVATIONS = {
        '0': ops.Identity(),
        '1': nn.ReLU(),
        '2': nn.Sigmoid(),
        '3': nn.Tanh(),
    }

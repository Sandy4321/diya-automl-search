from copy import deepcopy
import torch.nn as nn
import genotype


class MixedOp(nn.Module):
    def __init__(self, cell, size):
        super().__init__()
        self.ops = nn.ModuleList()
        for op in cell.OPERATIONS.values():
            for activation in cell.ACTIVATIONS.values():
                full_op = nn.Sequential(
                    getattr(genotype.operations, op)(size),
                    activation
                )
                self.ops.append(full_op)

    def forward(self, x, weights):
        return sum(w*op(x) for w, op in zip(weights, self.ops))


class Cell(nn.Module):
    def __init__(self, cell_type, size, n_nodes):
        super().__init__()
        if cell_type == 'cnn':
            self.type = genotype.cell.CNNCell
            self.preproc0 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(size[0], size[0], 3, padding=1),
            )

        elif cell_type == 'rnn':
            self.type = genotype.cell.RNNCell
            self.preproc0 = nn.Sequential(
                nn.ReLU(),
                nn.Conv1d(size[0], size[0], 1),
            )

        elif cell_type == 'transformer':
            self.type = genotype.cell.TransformerCell
            self.preproc0 = nn.Sequential(
                nn.ReLU(),
                nn.Conv1d(size[0], size[0], 1),
            )

        else:
            raise NotImplementedError

        self.preproc1 = deepcopy(self.preproc0)
        self.dag = nn.ModuleList()
        for i in range(n_nodes):
            self.dag.append(nn.ModuleList())
            for _ in range(i + 2):
                self.dag[i].append(MixedOp(self.type, size))

    def forward(self, s0, s1, w_agg, w_ops):
        s0 = self.preproc0(s0)
        s1 = self.preproc1(s1)
        states = [s0, s1]
        aggs = list(self.type.AGGREGATIONS.values())
        for node_idx, (edges, _w_ops) in enumerate(zip(self.dag, w_ops)):
            s_cur = 0
            for idx, _w_agg in enumerate(w_agg[node_idx]):
                if aggs[idx] == 'sum':
                    s_tmp = 0
                    for i, (s, w) in enumerate(zip(states, _w_ops)):
                        s_tmp = s_tmp + edges[i](s, w)
                elif aggs[idx] == 'product':
                    s_tmp = 1
                    for i, (s, w) in enumerate(zip(states, _w_ops)):
                        s_tmp = s_tmp*edges[i](s, w)
                else:
                    raise NotImplementedError
                s_cur = s_cur + _w_agg*s_tmp
            states.append(s_cur)
        return states[-1]

import genomes.operations as ops


class Node:
    IDX_SEP = 'i'
    OPERATIONS = {
        '0': 'sep_conv',
        '1': 'dil_conv',
        '2': 'max_pool',
        '3': 'avg_pool',
        '4': 'self_attn',
        '5': 'enc_attn',
        '6': 'identity',
        '7': 'none'
    }
    KERNEL_SIZE = {
        '0': 1,
        '1': 3,
        '2': 5
    }
    ACTIVATIONS = {
        '0': 'relu',
        '1': 'sigmoid',
        '2': 'tanh'
    }

    def __init__(self, seq):
        idx1, seq = seq.split(self.IDX_SEP, 1)
        ops1, seq = seq[:3], seq[3:]
        idx2, ops2 = seq.split(self.IDX_SEP, 1)
        self.idx1 = int(idx1)
        self.idx2 = int(idx2)

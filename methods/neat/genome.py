from neat.genome import DefaultGenome, DefaultGenomeConfig
from neat.genes import DefaultConnectionGene
from methods.neat.gene import CNNGene, RNNGene, TransformerGene
from genotype.cell import IDX_SEP


class CNNGenome(DefaultGenome):
    @classmethod
    def parse_config(cls, param_dict):
        param_dict['node_gene_type'] = CNNGene
        param_dict['connection_gene_type'] = DefaultConnectionGene
        return DefaultGenomeConfig(param_dict)


class RNNGenome(DefaultGenome):
    @classmethod
    def parse_config(cls, param_dict):
        param_dict['node_gene_type'] = RNNGene
        param_dict['connection_gene_type'] = DefaultConnectionGene
        return DefaultGenomeConfig(param_dict)


class TransformerGenome(DefaultGenome):
    @classmethod
    def parse_config(cls, param_dict):
        param_dict['node_gene_type'] = TransformerGene
        param_dict['connection_gene_type'] = DefaultConnectionGene
        return DefaultGenomeConfig(param_dict)


def model_info(genome):
    node_info = {}
    connection_info = {}
    for k in list(genome.nodes.keys()):
        node_info[k] = [
            genome.nodes[k].aggregation,
            genome.nodes[k].operation,
            genome.nodes[k].kernel_size,
            genome.nodes[k].activation
        ]
    for a, b in list(genome.connections.keys()):
        if b not in list(connection_info.keys()):
            connection_info[b] = [a]
        else:
            connection_info[b].append(a)
    return node_info, connection_info


def parse_info(node_idx, node_info, connection_info):
    in_idx1 = connection_info[2*node_idx] + 1
    out = [str(in_idx1), IDX_SEP]
    out += node_info[in_idx1][1:]
    if len(connection_info) % 2 == 0:
        in_idx2 = connection_info[2*node_idx + 1] + 1
        out += [str(in_idx2), IDX_SEP]
        out += node_info[in_idx2][1:]
    else:
        out += ['0', IDX_SEP, '6', '0', '0']
    return out


def make_genome(genome):
    node_info, conn_info = model_info(genome)
    index = sorted(list(conn_info.keys()))
    new_genome = []
    for idx in index:
        conn_info[idx] = sorted(conn_info[idx])
        if len(conn_info[idx]) == 1:
            agg = [node_info[idx][0]]
            in_idx1 = conn_info[idx][0] + 1
            ops = node_info[in_idx1][1:]
            ops = [str(in_idx1), IDX_SEP, *ops]
            ops += ['0', IDX_SEP, '6', '0', '0']
            new_genome.append(''.join(agg + ops))

        for node_idx in range(len(conn_info[idx]) // 2):
            agg = [node_info[idx][0]]
            ops = parse_info(node_idx, node_info, conn_info[idx])
            new_genome.append(''.join(agg + ops))

    return new_genome

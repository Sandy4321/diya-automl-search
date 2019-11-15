import networkx as nx
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
        if genome.connections[(a, b)].enabled:
            if b not in list(connection_info.keys()):
                connection_info[b] = [a]
            else:
                connection_info[b].append(a)
    return node_info, connection_info


def make_genome(genome):
    root = -2 if isinstance(genome, RNNGenome) else -1
    node_info, conn_info = model_info(genome)
    G = nx.DiGraph()
    for u, vs in conn_info.items():
        for v in vs:
            G.add_edge(v, u)
    if root not in G.nodes:
        index = []
    else:
        v_remove = []
        for v in G.nodes:
            if not nx.has_path(G, root, v) and v >= 0:
                v_remove.append(v)
        G.remove_nodes_from(v_remove)
        index = list(nx.topological_sort(G))

    new_genome = []
    for idx in index:
        if idx in conn_info:
            conn_info[idx] = sorted(conn_info[idx])
            for node_idx in range(len(conn_info[idx]) // 2):
                out = [node_info[idx][0]]
                in_idx1 = conn_info[idx][2*node_idx]
                if in_idx1 >= 0 and in_idx1 in index:
                    ops = node_info[in_idx1][1:]
                    in_idx1 = index.index(in_idx1)
                else:
                    if in_idx1 < 0:
                        ops = ['0', '0', '3']
                        in_idx1 = in_idx1 - root
                    else:
                        ops = ['6', '0', '0']
                        in_idx1 = 0
                out += [str(in_idx1), IDX_SEP, *ops]

                in_idx2 = conn_info[idx][2*node_idx + 1]
                if in_idx2 >= 0 and in_idx2 in index:
                    ops = node_info[in_idx2][1:]
                    in_idx2 = index.index(in_idx2)
                else:
                    if in_idx2 < 0:
                        ops = ['0', '0', '3']
                        in_idx2 = in_idx2 - root
                    else:
                        ops = ['6', '0', '0']
                        in_idx2 = 0
                out += [str(in_idx2), IDX_SEP, *ops]

                new_genome.append(''.join(out))

            if len(conn_info[idx]) % 2 == 1:
                out = [node_info[idx][0]]
                in_idx1 = conn_info[idx][-1]
                if in_idx1 >= 0 and in_idx1 in index:
                    ops = node_info[in_idx1][1:]
                    in_idx1 = index.index(in_idx1)
                else:
                    if in_idx1 < 0:
                        ops = ['0', '0', '3']
                        in_idx1 = in_idx1 - root
                    else:
                        ops = ['6', '0', '0']
                        in_idx1 = 0
                out += [str(in_idx1), IDX_SEP, *ops]
                out += [str(0), IDX_SEP, '6', '0', '0']

                new_genome.append(''.join(out))

    return new_genome

import networkx as nx
from networkx.drawing.nx_agraph import to_agraph
from genotype.cell import IDX_SEP
from genotype.cell import AGGREGATIONS, OPERATIONS

ACTIVATIONS = {
    '0': 'identity',
    '1': 'relu',
    '2': 'sigmoid',
    '3': 'tanh'
}


def draw_genome(cell_type, genome):
    G = nx.DiGraph()
    G.add_node(
        'input',
        style='filled',
        shape='square',
        fixedsize='true',
        width='0.8'
    )
    if cell_type == 'rnn':
        bias = 2
        G.add_node(
            'hidden',
            style='filled',
            shape='square',
            fixedsize='true',
            width='0.8'
        )
    else:
        bias = 1

    for idx, seq in enumerate(genome):
        agg, seq = seq[0], seq[1:]
        idx1, seq = seq.split(IDX_SEP, 1)
        op1, seq = seq[:3], seq[3:]
        idx2, op2 = seq.split(IDX_SEP, 1)

        count = 0
        if op1[:2] != '00':
            count += 1
            op1_name = str(idx1) + '_first_' + str(idx + bias)
            label = OPERATIONS[op1[:2]] + '\n' + ACTIVATIONS[op1[2]]
            G.add_node(
                op1_name,
                label=label,
                fixedsize='true',
                width='1.3',
                height='0.8'
            )
            if int(idx1) < bias:
                if int(idx1) == 0:
                    G.add_edge('input', op1_name)
                elif int(idx1) == 1:
                    G.add_edge('hidden', op1_name)
            else:
                agg_name = 'agg_' + str(idx1)
                if G.has_node(agg_name):
                    G.add_edge(agg_name, op1_name)
                else:
                    for node in G.nodes:
                        if '_' + str(idx1) in node:
                            G.add_edge(node, op1_name)

        if op2[:2] != '00':
            count += 1
            op2_name = str(idx2) + '_second_' + str(idx + bias)
            label = OPERATIONS[op2[:2]] + '\n' + ACTIVATIONS[op2[2]]
            G.add_node(
                op2_name,
                label=label,
                fixedsize='true',
                width='1.3',
                height='0.8'
            )
            if int(idx2) < bias:
                if int(idx2) == 0:
                    G.add_edge('input', op2_name)
                elif int(idx2) == 1:
                    G.add_edge('hidden', op2_name)
            else:
                agg_name = 'agg_' + str(idx2)
                if G.has_node(agg_name):
                    G.add_edge(agg_name, op2_name)
                else:
                    for node in G.nodes:
                        if '_' + str(idx2) in node:
                            G.add_edge(node, op2_name)

        if count == 2:
            agg_name = 'agg_' + str(idx + bias)
            if AGGREGATIONS[agg] == 'sum':
                G.add_node(agg_name, label='+', shape="circle")
            elif AGGREGATIONS[agg] == 'product':
                G.add_node(agg_name, label='x', shape="circle")
            G.add_edge(op1_name, agg_name)
            G.add_edge(op2_name, agg_name)

    last = '0'
    for node in G.nodes:
        try:
            if node[:3] == 'agg':
                if int(node[-1]) >= int(last[-1]):
                    last = node
            else:
                if int(node[-1]) > int(last[-1]):
                    last = node
        except ValueError:
            continue

    G.add_node(
        'output',
        style='filled',
        shape='square',
        fixedsize='true',
        width='0.8'
    )
    G.add_edge(last, 'output')

    to_remove = []
    for node in G.nodes:
        if not nx.has_path(G, node, 'output'):
            to_remove.append(node)
    G.remove_nodes_from(to_remove)

    G.graph['graph'] = {'rankdir': 'TD'}
    G.graph['node'] = {'shape': 'rect'}
    G.graph['edges'] = {'arrowsize': '4.0'}

    A = to_agraph(G)
    A.layout('dot')
    A.draw('genome.png')

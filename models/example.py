import os
import neat
from AutoML_genome import AutoML_Genome


def model_info(genome):
    node_info = {}
    connection_info = {}

    node_len = len(list(genome.nodes.keys()))
    for key in list(genome.nodes.keys()):
        if key == 0:
            node_info[-1] = [genome.nodes[0].operation[:-2], genome.nodes[0].operation[-1], genome.nodes[0].activation ]
            node_info[0] = [genome.nodes[0].operation[:-2], genome.nodes[0].operation[-1], genome.nodes[0].activation ]
        else:
            node_info[key] = [genome.nodes[key].operation[:-2], genome.nodes[key].operation[-1], genome.nodes[key].activation ]
    for a,b in list(genome.connections.keys()): 
        if b not in list(connection_info.keys()):
            connection_info[b] = [a]
        else:
            connection_info[b].append(a)
    return node_info, connection_info
def make_genome(genome):
    node_info, connection_info = model_info(genome)
    index = sorted(list(connection_info.keys()))
    new_genome = []
    for idx in index:
      connection_info[idx] = sorted(connection_info[idx])
   
      for node_idx in range(len(connection_info[idx]) // 2):
        in_idx1 = connection_info[idx][2*node_idx]
        in_idx2 = connection_info[idx][2*node_idx + 1]
        new_genome.append([in_idx1,'i',node_info[in_idx1][0],node_info[in_idx1][1],node_info[in_idx1][2], in_idx2,'i',node_info[in_idx2][0],node_info[in_idx2][1],node_info[in_idx2][2]])        

      if len(connection_info[idx]) % 2  == 1 :
        in_idx1 = connection_info[idx][-1]
        new_genome.append([in_idx1,'i',node_info[in_idx1][0],node_info[in_idx1][1],node_info[in_idx1][2], 'none', 'i', 'none', 'none', 'none'])
    return new_genome


# 2-input XOR inputs and expected outputs.
xor_inputs = [(0.0,), (0.0,), (1.0,), (1.0,)]
xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]



def model_info(genome):
    node_info = {}
    connection_info = {}

    node_len = len(list(genome.nodes.keys()))
    for key in list(genome.nodes.keys()) :
        if key == 0:
            node_info[-1] = [genome.nodes[0].operation[:-2], genome.nodes[0].operation[-1], genome.nodes[0].activation ]
        else:
            node_info[key] = [genome.nodes[key].operation[:-2], genome.nodes[key].operation[-1], genome.nodes[key].activation ]
    for a,b in list(genome.connections.keys()) : 
        if b not in list(connection_info.keys()):
            connection_info[b] = [a]
        else:
            connection_info[b].append(a)
    return node_info, connection_info
def make_genome(genome):
    node_info, connection_info = model_info(genome)
    index = sorted(list(connection_info.keys()))
    new_genome = []
    for idx in index:
        if len(connection_info[idx]) >= 2 :
            connection_info[idx] = sorted(connection_info[idx])[-2:]
            in_idx1 = connection_info[idx][0]
            in_idx2 = connection_info[idx][1]
            new_genome.append([in_idx1,'i',node_info[in_idx1][0],node_info[in_idx1][1],node_info[in_idx1][2], in_idx2,'i',node_info[in_idx2][0],node_info[in_idx2][1],node_info[in_idx2][2]])
        else:
            in_idx1 = connection_info[idx][0]
            new_genome.append([in_idx1,'i',node_info[in_idx1][0],node_info[in_idx1][1],node_info[in_idx1][2], 'none', 'i', 'none', 'none', 'none'])
    return new_genome



# 2-input XOR inputs and expected outputs.
xor_inputs = [(0.0,), (0.0,), (1.0,), (1.0,)]
xor_outputs = [   (0.0,),     (1.0,),     (1.0,),     (0.0,)]


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for xi, xo in zip(xor_inputs, xor_outputs):
            output = net.activate(xi)
            genome.fitness -= (output[0] - xo[0]) ** 2


'''
def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 4.0
<<<<<<< HEAD
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for xi, xo in zip(xor_inputs, xor_outputs):
            output = net.activate(xi)
            genome.fitness -= (output[0] - xo[0]) ** 2


'''
def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 4.0
=======
>>>>>>> Genome making function
        genome = make_genome(genome)
        # change genome to child model and calculate the fitness of the model 
        # genome.fitness 추가해야함 
'''


def run(config_file):
    # Load configuration.
    config = neat.Config(AutoML_Genome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Run for up to 2 generations.
    winner = p.run(eval_genomes, 100)
    print(winner)
    return winner, config


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'example.txt')
    winner, config = run(config_path)
    node_info, connection_info = model_info(winner)
    winner_genome = make_genome(winner)
    print(node_info)
    print(connection_info)
    print(winner_genome)


"""
output example

Key: 4470
Fitness: 2.999999823901414
Nodes:
        0 AutoML_NodeGene(key=0, bias=0.03849047298552355, response=1.0, aggregation=sum, activation=sigmoid, operation=attention_1)
        15 AutoML_NodeGene(key=15, bias=1.7205519859849985, response=1.0, aggregation=sum, activation=tanh, operation=max_pool_3)
        24 AutoML_NodeGene(key=24, bias=-0.04670113893579997, response=1.0, aggregation=sum, activation=tanh, operation=dil_conv_3)
        108 AutoML_NodeGene(key=108, bias=1.795003183765722, response=1.0, aggregation=sum, activation=sigmoid, operation=dil_conv_1)
        115 AutoML_NodeGene(key=115, bias=1.0961638523379467, response=1.0, aggregation=sum, activation=identity, operation=avg_pool_5)
        913 AutoML_NodeGene(key=913, bias=-0.7451362263638321, response=1.0, aggregation=sum, activation=sigmoid, operation=attention_1)
Connections:
        DefaultConnectionGene(key=(-1, 24), weight=0.7522124101303344, enabled=True)
        DefaultConnectionGene(key=(-1, 108), weight=2.209572204762623, enabled=False)
        DefaultConnectionGene(key=(-1, 913), weight=0.3763350361429767, enabled=True)
        DefaultConnectionGene(key=(15, 0), weight=-0.03866826369667824, enabled=True)
        DefaultConnectionGene(key=(24, 15), weight=0.5016280260590973, enabled=True)
        DefaultConnectionGene(key=(24, 108), weight=2.0496015569781747, enabled=True)
        DefaultConnectionGene(key=(115, 108), weight=1.7574174386898913, enabled=True)
        DefaultConnectionGene(key=(913, 108), weight=1.4426047149465215, enabled=True)
{-1: ['attention', '1', 'sigmoid'], 0: ['attention', '1', 'sigmoid'], 15: ['max_pool', '3', 'tanh'], 24: ['dil_conv', '3', 'tanh'], 108: ['dil_conv', '1', 'sigmoid'], 115: ['avg_pool', '5', 'identity'], 913: ['attention', '1', 'sigmoid']}
{0: [15], 24: [-1], 15: [24], 108: [24, 115, -1, 913], 913: [-1]}
[[15, 'i', 'max_pool', '3', 'tanh', 'none', 'i', 'none', 'none', 'none'], [24, 'i', 'dil_conv', '3', 'tanh', 'none', 'i', 'none', 'none', 'none'], [-1, 'i', 'attention', '1', 'sigmoid', 
'none', 'i', 'none', 'none', 'none'], [-1, 'i', 'attention', '1', 'sigmoid', 24, 'i', 'dil_conv', '3', 'tanh'], [115, 'i', 'avg_pool', '5', 'identity', 913, 'i', 'attention', '1', 'sigmoid'], [-1, 'i', 'attention', '1', 'sigmoid', 'none', 'i', 'none', 'none', 'none']]
"""

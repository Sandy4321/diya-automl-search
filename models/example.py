import os
import neat
from AutoML_genome import AutoML_Genome
from itertools import count

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 4.0
        # change genome to child model and calculate the fitness of the model 
        # genome.fitness 추가해야함 

def run(config_file):
    # Load configuration.
    config = neat.Config(AutoML_Genome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Run for up to 2 generations.
    winner = p.run(eval_genomes, 2)
    print(winner)
    return winner, config

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'example.txt')
    run(config_path)
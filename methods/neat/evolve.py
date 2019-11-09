import os
import neat
import numpy as np
import torch.nn as nn

from settings import PROJECT_ROOT
from utils.summary import AverageMeter
from trainer import Trainer
from genotype.cell import CNNCell, RNNCell, TransformerCell
from genotype.network import FeedForward, Recurrent
from methods.base import Base
from methods.neat.genome import CNNGenome, RNNGenome, TransformerGenome
from methods.neat.genome import make_genome


class NEAT(Base):
    def __init__(self, genome_type, env, args):
        super().__init__('NEAT', args)
        if genome_type == 'cnn':
            genome = CNNGenome
            self.size = tuple([args.dim, *env['size'][1:]])
            self.stem = nn.Conv2d(env['size'][0], args.dim, 3, padding=1)
            self.cell = CNNCell
            self.classifier = nn.Linear(
                args.dim*np.prod(env['size'][1:]),
                env['num_classes']
            )
            self.network = FeedForward
        elif genome_type == 'rnn':
            genome = RNNGenome
            self.size = tuple([args.dim])
            self.stem = nn.Linear(env['size'][1], args.dim // 2)
            self.cell = RNNCell
            self.classifier = nn.Linear(
                args.dim // 2,
                env['num_classes']
            )
            self.network = Recurrent
        elif genome_type == 'transformer':
            genome = TransformerGenome
            self.size = tuple([env['size'][0], args.dim])
            self.stem = nn.Linear(env['size'][1], args.dim)
            self.cell = TransformerCell
            self.classifier = nn.Linear(
                args.dim*env['size'][0],
                env['num_classes']
            )
            self.network = FeedForward
        else:
            raise NotImplementedError
        self.env = env
        self.args = args

        path = os.path.join(PROJECT_ROOT, 'methods', 'neat')
        config_file = os.path.join(path, 'config.txt')
        self.config = neat.Config(
            genome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_file
        )

    def eval_genomes(self, genomes, config):
        fitness = AverageMeter()
        for idx, (genome_id, genome) in enumerate(genomes):
            cell = self.cell(self.size, make_genome(genome))
            cells = nn.ModuleList([cell]*self.args.cells)
            model = self.network(self.stem, cells, self.classifier)

            trainer = Trainer(self.env['train'], model, self.args)
            for _ in range(self.args.epochs):
                trainer.train()
            trainer = Trainer(self.env['val'], model, self.args)
            trainer.infer()
            genome.fitness = trainer.info.avg['Accuracy/Top1']
            fitness.update(genome.fitness)
            if genome.fitness > self.best_fitness:
                self.best_fitness = genome.fitness
                self.logger.log("Saving genome at step {}...".format(
                    self.step
                ))
                filename = 'genome{}.txt'.format(self.step)
                path = os.path.join(self.logger.log_dir, filename)
                with open(path, 'w') as f:
                    seqs = make_genome(genome)
                    for seq in seqs:
                        f.write(seq)

            if self.step % self.args.log_step == 0:
                self.logger.log("Generation {}: {}/{} ({:.2f}%)".format(
                    self.generations,
                    idx + 1,
                    len(genomes),
                    (idx + 1)/len(genomes)*100
                ))
                info = {
                    'Generation': self.generations,
                    'Epochs': self.args.epochs*(idx + 1)*(self.step + 1),
                    'Accuracy/Avg': fitness.avg,
                    'Accuracy/Best': self.best_fitness
                }
                self.logger.scalar_summary(info, self.step)
                self.step += 1
        self.generations += 1

    def search(self):
        self.generations = 0
        self.step = 0
        self.best_fitness = 0
        self.logger.log("Begin evolution from population size: {}".format(
            self.config.pop_size)
        )
        p = neat.Population(self.config)
        _ = p.run(self.eval_genomes)

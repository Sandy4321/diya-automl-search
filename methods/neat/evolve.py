import os
import neat
from settings import PROJECT_ROOT
from utils.summary import AverageMeter
from utils.common import load_genome
from trainer import Trainer
import envs
from methods.base import Base
from methods.neat.genome import CNNGenome, RNNGenome, TransformerGenome
from methods.neat.genome import make_genome


class NEAT(Base):
    def __init__(self, args):
        super().__init__('NEAT', args)
        if args.type == 'cnn':
            genome = CNNGenome
        elif args.type == 'rnn':
            genome = RNNGenome
        elif args.type == 'transformer':
            genome = TransformerGenome
        else:
            raise NotImplementedError
        self.env = getattr(envs, args.env)(args)
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
            model = load_genome(
                make_genome(genome)[:self.args.nodes],
                self.args
            )
            trainer = Trainer(self.env, model, self.args)
            for _ in range(self.args.epochs):
                trainer.train()
            trainer.info.reset()
            trainer.infer(test=False)

            genome.fitness = trainer.info.avg['Accuracy/Top1']
            fitness.update(genome.fitness)
            if genome.fitness > self.best_fitness:
                self.best_fitness = genome.fitness
                self.logger.log("Saving genome at step {}...".format(
                    self.step
                ))
                filename = 'genome_{}.txt'.format(self.step)
                path = os.path.join(self.logger.log_dir, filename)
                with open(path, 'w') as f:
                    seqs = make_genome(genome)
                    for seq in seqs:
                        f.write(seq + '\n')

            if self.step % self.args.log_step == 0:
                self.logger.log("Generation {}: {}/{} ({:.2f}%)".format(
                    self.generations,
                    idx + 1,
                    len(genomes),
                    (idx + 1)/len(genomes)*100
                ))
                info = {
                    'Generation': self.generations,
                    'Epochs': self.args.epochs*(self.step + 1),
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

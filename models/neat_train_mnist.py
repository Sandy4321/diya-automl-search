import os
import neat
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch

# 2-input XOR inputs and expected outputs.

class MNIST(datasets.MNIST):
    def __init__(self, root, train):
        super().__init__(root, train=train, download=False)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3081])
        ])
def mnist():
    root = os.path.join('C:/Users/djaej_000/Desktop/neat', 'mnist')
    return {
        'size': (1, 28, 28),
        'num_classes': 10,
        'train': DataLoader(
            MNIST(root, train=True),
            batch_size=1,
            num_workers=0,
            drop_last=True
        ),
        'test': DataLoader(
            MNIST(root, train=False),
            batch_size=1,
            num_workers=0,
            drop_last=False
        )
    }


class loader:
    def __init__(self, mode = 'train'):
        self.mode = mode  # or 'test'
        self.mnist_loader = mnist()
        self.loss = torch.nn.CrossEntropyLoss()
    def eval_genomes(self,genomes,  config):
        for genome_id, genome in genomes:
            genome.fitness = 0
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            loader = self.mnist_loader[self.mode]
            total_batch = len(loader)
            print('{} : total batch | {} : genome_id '.format(total_batch,genome_id))
            for i, batch in enumerate(loader):
                inputs, targets = batch
                inputs = inputs.view(-1).tolist()
                output = net.activate(inputs)
                output = torch.tensor(output).unsqueeze(0)
                genome.fitness -= self.loss(output,targets).item()
                if i % 100 == 0 :
                    print('%0.2f %% done'%(i*100 / total_batch))
            genome.fitness = genome.fitness/total_batch


def run(config_file, is_train = True):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    
    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)
    neat_loader = loader(mode = 'train')

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(neat_loader.eval_genomes,10)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    neat_loader.mode = 'test'
    p.run(neat_loader.eval_genomes, 10)
    return winner, config

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat_config_cnn.txt')
    run(config_path)

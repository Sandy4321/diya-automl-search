import os
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn

from settings import PROJECT_ROOT
from utils.common import ArgumentParser
from utils.logger import Logger
from trainer import Trainer


def resnet(args):
    from envs import mnist
    from genotype.cell import CNNCell
    from genotype.network import FeedForward

    args.split_ratio = 1.0
    env = mnist(args)
    path = os.path.join(PROJECT_ROOT, 'genotype', 'examples')
    filename = os.path.join(path, 'resnet.txt')
    with open(filename, 'r') as f:
        genome = f.readlines()

    size = tuple([args.dim, *env['size'][1:]])
    stem = nn.Conv2d(env['size'][0], args.dim, 3, padding=1)
    cells = nn.ModuleList([CNNCell(size, genome)]*args.cells)
    classifier = nn.Linear(
        args.dim*np.prod(env['size'][1:]),
        env['num_classes']
    )
    model = FeedForward(stem, cells, classifier)

    logger = Logger('MAIN', args=args)
    logger.log("Begin training resnet-like network")
    trainer = Trainer(env['train'], model, args)
    for epoch in range(args.epochs):
        trainer.train()
        if epoch % args.log_step == 0:
            logger.log("Training statistics for epoch: {}".format(epoch))
            logger.scalar_summary(trainer.info.avg, epoch)
            trainer.info.reset()

    trainer = Trainer(env['test'], model, args)
    trainer.infer()
    logger.log("Validation accuracy: {}".format(
        trainer.info.avg['Accuracy/Top1']
    ))


def lstm(args):
    from envs import imdb_glove50d
    from genotype.cell import RNNCell
    from genotype.network import Recurrent

    args.split_ratio = 0.99
    env = imdb_glove50d(args)
    path = os.path.join(PROJECT_ROOT, 'genotype', 'examples')
    filename = os.path.join(path, 'lstm.txt')
    with open(filename, 'r') as f:
        genome = f.readlines()

    size = tuple([args.dim, 1])
    stem = nn.Linear(env['size'][1], args.dim // 2)
    cells = nn.ModuleList([RNNCell(size, genome)]*args.cells)
    classifier = nn.Linear(
                args.dim // 2,
                env['num_classes']
            )
    model = Recurrent(stem, cells, classifier)

    logger = Logger('MAIN', args=args)
    logger.log("Begin training lstm-like network")
    trainer = Trainer(env['train'], model, args)
    for epoch in range(args.epochs):
        trainer.train()
        if epoch % args.log_step == 0:
            logger.log("Training statistics for epoch: {}".format(epoch))
            logger.scalar_summary(trainer.info.avg, epoch)
            trainer.info.reset()

    trainer = Trainer(env['test'], model, args)
    trainer.infer()
    logger.log("Validation accuracy: {}".format(
        trainer.info.avg['Accuracy/Top1']
    ))


def transformer(args):
    from envs import imdb_glove50d
    from genotype.cell import TransformerCell
    from genotype.network import FeedForward

    args.split_ratio = 0.99
    env = imdb_glove50d(args)
    path = os.path.join(PROJECT_ROOT, 'genotype', 'examples')
    filename = os.path.join(path, 'transformer.txt')
    with open(filename, 'r') as f:
        genome = f.readlines()

    size = tuple([env['size'][0], args.dim])
    stem = nn.Linear(env['size'][1], args.dim)
    cells = nn.ModuleList([TransformerCell(size, genome)]*args.cells)
    classifier = nn.Linear(
        args.dim*env['size'][0],
        env['num_classes']
    )
    model = FeedForward(stem, cells, classifier)

    logger = Logger('MAIN', args=args)
    logger.log("Begin training transformer-like network")
    trainer = Trainer(env['train'], model, args)
    for epoch in range(args.epochs):
        trainer.train()
        if epoch % args.log_step == 0:
            logger.log("Training statistics for epoch: {}".format(epoch))
            logger.scalar_summary(trainer.info.avg, epoch)
            trainer.info.reset()

    trainer = Trainer(env['test'], model, args)
    trainer.infer()
    logger.log("Validation accuracy: {}".format(
        trainer.info.avg['Accuracy/Top1']
    ))


def neat_cnn(args):
    from envs import mnist
    from methods.neat import NEAT

    env = mnist(args)
    neat = NEAT('cnn', env, args)
    neat.search()


def neat_rnn(args):
    from envs import imdb_glove50d
    from methods.neat import NEAT

    env = imdb_glove50d(args)
    neat = NEAT('rnn', env, args)
    neat.search()


def neat_transformer(args):
    from envs import imdb_glove50d
    from methods.neat import NEAT

    env = imdb_glove50d(args)
    neat = NEAT('transformer', env, args)
    neat.search()


def test(args):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoML Meetup")
    parser.add_argument("--load_config", type=str)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--mode", type=str, default='test')
    parser.add_argument("--seed", type=str, default=100)

    parser.add_argument_group("logger options")
    parser.add_argument("--log_level", type=int, default=20)
    parser.add_argument("--log_step", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_true")

    parser.add_argument_group("dataset options")
    parser.add_argument("--env", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument_group("search options")
    parser.add_argument("--method", type=str)
    parser.add_argument("--split_ratio", type=float, default=0.5)
    parser.add_argument("--nodes", type=int, default=4)
    parser.add_argument("--cells", type=int, default=4)
    parser.add_argument("--dim", type=int, default=32)

    parser.add_argument_group("training options")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)

    args = parser.parse_args()
    if args.load_config is not None:
        with open(os.path.join(PROJECT_ROOT, args.load_config)) as config:
            args = ArgumentParser(json.load(config))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.tag is None:
        args.tag = args.mode
    if args.debug:
        args.log_level = 1
    elif args.quiet:
        args.log_level = 30
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    globals()[args.mode](args)

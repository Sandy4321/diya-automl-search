import os
import json
import argparse
import random
import numpy as np
import torch
from settings import PROJECT_ROOT
from utils.common import ArgumentParser


def neat_cnn(args):
    from envs import mnist
    from methods.neat import NEAT

    env = mnist(args)
    neat = NEAT('cnn', env, args)
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
    parser.add_argument("--lr", type=float, default=0.01)
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

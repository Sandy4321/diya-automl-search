import os
import json
import argparse
import random
import numpy as np
import torch
from settings import PROJECT_ROOT, LOAD_DIR
from utils.common import ArgumentParser, load_model
import envs
import models
from trainer import Trainer


def test(args):
    from genomes.genome import Genome
    Genome('genomes/lenet.txt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoML Meetup")
    parser.add_argument("--load_config", type=str)

    parser.add_argument("--tag", type=str, default='test')
    parser.add_argument("--mode", type=str, default='test')
    parser.add_argument("--seed", type=str, default=100)

    parser.add_argument_group("logger options")
    parser.add_argument("--log_level", type=int, default=20)
    parser.add_argument("--log_step", type=int, default=100)
    parser.add_argument("--debug", action="store_true")

    parser.add_argument_group("dataset options")
    parser.add_argument("--env", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument_group("training options")
    parser.add_argument("--model", type=str)
    parser.add_argument("--checkpoint", type=str)

    args = parser.parse_args()
    if args.load_config is not None:
        with open(os.path.join(PROJECT_ROOT, args.load_config)) as config:
            args = ArgumentParser(json.load(config))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.debug:
        args.log_level = 1

    globals()[args.mode](args)
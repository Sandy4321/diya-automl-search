import os
import json
import argparse
import random
import numpy as np
import torch

import settings
import utils.common as common
from utils.logger import Logger
import envs
import methods
from trainer import Trainer


def search(args):
    assert args.method is not None
    method = getattr(methods, args.method)(args)
    method.search()


def train(args):
    assert args.checkpoint is not None
    path = os.path.join(settings.PROJECT_ROOT, settings.LOAD_DIR)
    if args.checkpoint[-3:] == 'txt':
        with open(os.path.join(path, args.checkpoint), 'r') as f:
            genome = f.readlines()
            model = common.load_genome(genome, args)
    else:
        model = common.load_model(os.path.join(path, args.checkpoint))

    args.split_ratio = 0.99
    env = getattr(envs, args.env)(args)
    trainer = Trainer(env, model, args)

    logger = Logger('MAIN', args=args)
    logger.log("Begin training {}".format(args.checkpoint))
    best_acc = 0
    for epoch in range(args.epochs):
        trainer.train()
        if epoch % args.log_step == 0:
            logger.log("Training statistics for epoch: {}".format(epoch))
            logger.scalar_summary(trainer.info.avg, epoch)
            trainer.info.reset()

        trainer.infer(test=True)
        acc = trainer.info.avg['Accuracy/Top1']
        trainer.info.reset()
        logger.log("Validation accuracy: {}".format(acc))
        if acc > best_acc:
            best_acc = acc
            path = os.path.join(logger.log_dir, 'model.pth'.format(epoch))
            logger.log("Saving model at epoch: {}".format(epoch))
            common.save_model(model, path)


def draw(args):
    from utils.visualize import draw_genome
    assert args.checkpoint is not None
    assert args.checkpoint[-3:] == 'txt'

    path = os.path.join(settings.PROJECT_ROOT, settings.LOAD_DIR)
    with open(os.path.join(path, args.checkpoint), 'r') as f:
        genome = f.readlines()
    draw_genome(args.type, genome)


def test(args):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoML Meetup")
    parser.add_argument("--load_config", type=str)
    parser.add_argument("--tag", type=str)
    parser.add_argument("--mode", type=str, default='test')
    parser.add_argument("--seed", type=str, default=100)

    parser.add_argument_group("logger options")
    parser.add_argument("--log_level", type=int, default=20)
    parser.add_argument("--log_step", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_true")

    parser.add_argument_group("dataset options")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument_group("search options")
    parser.add_argument("--method", type=str.upper)
    parser.add_argument("--type", type=str)
    parser.add_argument("--split_ratio", type=float, default=0.5)
    parser.add_argument("--nodes", type=int, default=4)
    parser.add_argument("--cells", type=int, default=4)
    parser.add_argument("--dim", type=int, default=64)

    parser.add_argument_group("training options")
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--env", type=str)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)

    args = parser.parse_args()
    if args.load_config is not None:
        path = os.path.join(settings.PROJECT_ROOT, args.load_config)
        with open(path) as config:
            args = common.ArgumentParser(json.load(config))

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

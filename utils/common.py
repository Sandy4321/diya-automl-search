import torch
import torch.nn as nn
import numpy as np
import envs
import genotype


class ArgumentParser(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def save_model(model, path):
    pth = {
        'model': model,
        'checkpoint': model.state_dict(),
    }
    torch.save(pth, path)


def load_model(path):
    pth = torch.load(path, map_location=lambda storage, loc: storage)
    model = pth['model']
    model.load_state_dict(pth['checkpoint'])
    return model


def load_genome(genome, args):
    assert args.type is not None
    env = getattr(envs, args.env)(args)
    if args.type == 'cnn':
        size = tuple([args.dim, *env['size'][1:]])
        stem = nn.Sequential(
            nn.Conv2d(env['size'][0], args.dim, 3, padding=1),
            nn.Dropout2d(args.dropout)
        )
        cells = nn.ModuleList(
            [genotype.cell.CNNCell(size, genome) for _ in range(args.cells)]
        )
        classifier = nn.Sequential(
            nn.Dropout(args.dropout),
            nn.Linear(
                args.dim*np.prod(env['size'][1:]),
                env['num_classes']
            )
        )
        return genotype.network.FeedForward(stem, cells, classifier)

    elif args.type == 'rnn':
        size = tuple([args.dim, 1])
        stem = nn.Sequential(
            nn.Linear(env['size'][1], args.dim),
            nn.Dropout(args.dropout)
        )
        cell = genotype.cell.RNNCell(size, genome)
        classifier = nn.Sequential(
            nn.Dropout(args.dropout),
            nn.Linear(
                args.dim,
                env['num_classes']
            )
        )
        return genotype.network.Recurrent(stem, cell, classifier)

    elif args.type == 'transformer':
        size = tuple([env['size'][0], args.dim])
        stem = nn.Sequential(
            nn.Linear(env['size'][1], args.dim),
            nn.Dropout(args.dropout)
        )
        cells = nn.ModuleList(
            ([genotype.cell.TransformerCell(size, genome)
             for _ in range(args.cells)])
        )
        classifier = nn.Sequential(
            nn.Dropout(args.dropout),
            nn.Linear(
                args.dim*env['size'][0],
                env['num_classes']
            )
        )
        return genotype.network.FeedForward(stem, cells, classifier)

    else:
        raise NotImplementedError

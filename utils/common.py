import torch
import numpy as np


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


def get_params(model):
    return filter(lambda p: p.requires_grad, model.parameters())


def get_bytes(model):
    total = 0
    for param in model.parameters():
        size = np.prod(param.size())
        total += 4*size
    return total

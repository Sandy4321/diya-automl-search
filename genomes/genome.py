import torch.nn as nn
from genomes.operations import OPS_1D, OPS_2D


class Genome:
    def __init__(self, file):
        # TODO:
        # create different genome classes for CNN, RNN, Transformer

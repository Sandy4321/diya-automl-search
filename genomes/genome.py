import torch.nn as nn
from operations import OPS_1D, OPS_2D


class Genome:
    def __init__(self, file):
        with open(file, 'r') as f:
            lines = f.readlines()
        for l in lines:
            print(l)
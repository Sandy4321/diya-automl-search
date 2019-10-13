import warnings
from random import random
from neat.attributes import FloatAttribute, BoolAttribute, StringAttribute
from neat.genes import BaseGene


class AutoML_NodeGene(BaseGene):
    _gene_attributes = [FloatAttribute('bias'),  
                        FloatAttribute('response'),
                        StringAttribute('aggregation', options='sum'),
                        StringAttribute('activation', options=['none','identity','tanh','relu','sigmoid']),
                        StringAttribute('operation',options = ['none','skip_connect','sep_conv_3x3','sep_conv_5x5','sep_conv_7x7','dil_conv_3x3' ,'dil_conv_5x5' ,'avg_pool_3x3','max_pool_3x3'])
                        ]

    def __init__(self, key):
        assert isinstance(key, int), "DefaultNodeGene key must be an int, not {!r}".format(key)
        BaseGene.__init__(self, key)

    def distance(self, other, config):
        d = abs(self.bias - other.bias) + abs(self.response - other.response)
        if self.activation != other.activation:
            d += 1.0
        if self.operation != other.operation:
            d += 1.0
        return d * config.compatibility_weight_coefficient

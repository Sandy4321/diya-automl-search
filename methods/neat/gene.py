from neat.genes import BaseGene
from neat.attributes import StringAttribute
from genotype.cell import CNNCell, RNNCell, TransformerCell


def get_attributes(cell):
    # Remove 'none' from operations
    operations = list(cell.OPERATIONS.keys())
    if '6' in operations:
        operations.remove('6')
    return [
        StringAttribute('aggregation', options=cell.AGGREGATIONS.keys()),
        StringAttribute('operation', options=operations),
        StringAttribute('kernel_size', options=cell.KERNEL_SIZE.keys()),
        StringAttribute('activation', options=cell.ACTIVATIONS.keys()),
    ]


class CNNGene(BaseGene):
    def __init__(self, key):
        super().__init__(key)
        self._gene_attributes = get_attributes(CNNCell)

    def distance(self, other, config):
        d = super().distance(other, config)
        if (self.operation != other.operation or
                self.kernel_size != other.kernel_size):
            d += 1.0*config.compatibility_weight_coefficient
        return d


class RNNGene(BaseGene):
    def __init__(self, key):
        super().__init__(key)
        self._gene_attributes = get_attributes(RNNCell)

    def distance(self, other, config):
        d = super().distance(other, config)
        if (self.aggregation != other.aggregation or
                self.operation != other.operation):
            d += 1.0*config.compatibility_weight_coefficient
        return d


class TransformerGene(BaseGene):
    def __init__(self, key):
        super().__init__(key)
        self._gene_attributes = get_attributes(TransformerCell)

    def distance(self, other, config):
        d = super().distance(other, config)
        if (self.operation != other.operation or
                self.kernel_size != other.kernel_size):
            d += 1.0*config.compatibility_weight_coefficient
        return d

from neat.genes import BaseGene
from neat.attributes import StringAttribute
from genotype.cell import CNNCell, RNNCell, TransformerCell


def get_attributes(cell):
    # Remove 'none' from operations
    operations = list(cell.OPERATIONS.keys())
    if '00' in operations:
        operations.remove('00')
    return [
        StringAttribute('aggregation', options=cell.AGGREGATIONS.keys()),
        StringAttribute('operation', options=operations),
        StringAttribute('activation', options=cell.ACTIVATIONS.keys()),
    ]


class CNNGene(BaseGene):
    _gene_attributes = get_attributes(CNNCell)

    def distance(self, other, config):
        if self.operation != other.operation:
            return 1.0*config.compatibility_weight_coefficient
        else:
            return 0.0


class RNNGene(BaseGene):
    _gene_attributes = get_attributes(RNNCell)

    def distance(self, other, config):
        if (self.aggregation != other.aggregation or
                self.operation != other.operation):
            return 1.0*config.compatibility_weight_coefficient
        else:
            return 0.0


class TransformerGene(BaseGene):
    _gene_attributes = get_attributes(TransformerCell)

    def distance(self, other, config):
        if (self.operation != other.operation or
                self.activation != other.activation):
            return 1.0*config.compatibility_weight_coefficient
        else:
            return 0.0

from itertools import count
from random import choice, random, shuffle

import sys

from neat.aggregations import AggregationFunctionSet
from neat.config import ConfigParameter, write_pretty_params
from neat.genes import DefaultConnectionGene
from neat.graphs import creates_cycle
from neat.six_util import iteritems, iterkeys
from AutoML_gene import AutoML_NodeGene
from neat.genome import DefaultGenomeConfig, DefaultGenome
from operations_neat import OperationFunctionSet
from activations_neat import ActivationFunctionSet

class AutoML_GenomeConfig(DefaultGenomeConfig):
    def __init__(self,params):
        super(AutoML_GenomeConfig,self).__init__(params)

        # create operation sets
        self.activation_defs = ActivationFunctionSet()
        self.operation_defs = OperationFunctionSet()
        
    def add_operation(self, name, func):
        self.operation_defs.add(name, func)
    def add_activation(self, name, func):
        self.activation_defs.add(name, func)

class AutoML_Genome(DefaultGenome): #Genome type

    @classmethod
    def parse_config(cls, param_dict):
        param_dict['node_gene_type'] = AutoML_NodeGene
        param_dict['connection_gene_type'] = DefaultConnectionGene
        return AutoML_GenomeConfig(param_dict)
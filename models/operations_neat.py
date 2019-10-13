from operations import *
import math
import types



class InvalidOperationFunction(TypeError):
    pass


def validate_operation(function):
    if not isinstance(function,
                      (types.BuiltinFunctionType,
                       types.FunctionType,
                       types.LambdaType)):
        raise InvalidOperationFunction("A operation object is required.")

    #if function.__code__.co_argcount != 1:  # avoid deprecated use of `inspect`
    #   raise InvalidOperationFunction("A single-argument operation is required.")


class OperationFunctionSet(object):
    """
    Contains the list of current valid operation functions,
    including methods for adding and getting them.
    """

    def __init__(self):
        self.functions = {}
        self.add('none', OPS_2D['none'])
        self.add('skip_connect', OPS_2D['skip_connect'])
        self.add('sep_conv_3x3', OPS_2D['sep_conv_3x3'])
        self.add('sep_conv_5x5', OPS_2D['sep_conv_5x5'])
        self.add('sep_conv_7x7', OPS_2D['sep_conv_7x7'])
        self.add('dil_conv_3x3', OPS_2D['dil_conv_3x3'])
        self.add('dil_conv_5x5', OPS_2D['dil_conv_5x5'])
        self.add('avg_pool_3x3', OPS_2D['avg_pool_3x3'])
        self.add('max_pool_3x3', OPS_2D['max_pool_3x3'])

    def add(self, name, function):
        validate_operation(function)
        self.functions[name] = function

    def get(self, name):
        f = self.functions.get(name)
        if f is None:
            raise InvalidOperationFunction("No such operation function: {0!r}".format(name))

        return f

    def is_valid(self, name):
        return name in self.functions
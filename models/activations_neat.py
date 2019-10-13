import math
import types



def sigmoid_activation(z):
    z = max(-60.0, min(60.0, 5.0 * z))
    return 1.0 / (1.0 + math.exp(-z))


def tanh_activation(z):
    z = max(-60.0, min(60.0, 2.5 * z))
    return math.tanh(z)


def sin_activation(z):
    z = max(-60.0, min(60.0, 5.0 * z))
    return math.sin(z)


def gauss_activation(z):
    z = max(-3.4, min(3.4, z))
    return math.exp(-5.0 * z ** 2)


def relu_activation(z):
    return z if z > 0.0 else 0.0


def identity_activation(z):
    return z

def none_activation(z):
    return 0.0

class InvalidActivationFunction(TypeError):
    pass


def validate_activation(function):
    if not isinstance(function,
                      (types.BuiltinFunctionType,
                       types.FunctionType,
                       types.LambdaType)):
        raise InvalidActivationFunction("A function object is required.")

    if function.__code__.co_argcount != 1:  # avoid deprecated use of `inspect`
        raise InvalidActivationFunction("A single-argument function is required.")


class ActivationFunctionSet(object):
    """
    Contains the list of current valid activation functions,
    including methods for adding and getting them.
    """

    def __init__(self):
        self.functions = {}
        self.add('sigmoid', sigmoid_activation)
        self.add('tanh', tanh_activation)
        self.add('relu', relu_activation)
        self.add('identity', identity_activation)
        self.add('none', none_activation)

    def add(self, name, function):
        validate_activation(function)
        self.functions[name] = function

    def get(self, name):
        f = self.functions.get(name)
        if f is None:
            raise InvalidActivationFunction("No such activation function: {0!r}".format(name))

        return f

    def is_valid(self, name):
        return name in self.functions
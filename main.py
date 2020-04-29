import math

from network import Neuron, Layer, Network


def sigmoid(x: float) -> float:
    """The sigmoid activation function."""
    return 1 / (1 + math.e ** -x)


def make_neuron() -> Neuron:
    """Make a neuron.

    It will have a bias of zero and weights of 0.5. It will accept two inputs
    and use the sigmoid activation function.
    """
    return Neuron((0.5, 0.5), 0, sigmoid)


def make_layer(neurons: int) -> Layer:
    """Make a layer with some amount of neurons.

    Neurons will have a bias of zero and weights of 0.5, and will use the
    sigmoid activation function. They will have two inputs.
    """
    return Layer(make_neuron() for _ in range(neurons))


def make_network() -> Network:
    """Make a network.

    It will have two layers, the first with two neurons and the second with
    just one. Biases will start at zero and weights at 0.5. It will use the
    sigmoid activation function.
    """
    return Network((make_layer(2), make_layer(1)))

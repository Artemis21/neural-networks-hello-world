"""The network and components thereof."""
import typing


# just to make typing (in both senses) less repetetive
array = typing.Iterable[float]


def dot_product(a: array, b: array) -> float:
    """Find the dot product of a and b as if they were 1d arrays."""
    return sum(i * j for i, j in zip(a, b))


class Neuron:
    """A neuron in a layer of the network."""

    def __init__(self, weights: array, bias: float,
                 activation: typing.Callable):
        """Create a new neuron."""
        self.weights = weights
        self.bias = bias
        self.activation = activation

    def feed_forward(self, inputs: array) -> float:
        """Work out output based on an input."""
        return self.activation(dot_product(self.weights, inputs) + self.bias)


class Layer:
    """A layer in the network."""

    def __init__(self, neurons: array):
        """Create a new layer."""
        self.neurons = neurons

    def feed_forward(self, inputs: array) -> array:
        """Work out the layer's output for the previous layer's output."""
        return [neuron.feed_forward(inputs) for neuron in self.neurons]


class Network:
    """The neural network."""

    def __init__(self, layers: typing.Iterable[Layer]):
        """Create a neural new network."""
        self.layers = layers

    def feed_forward(self, inputs: array) -> array:
        """Work out the network's output for some inputs."""
        for layer in self.layers:
            inputs = layer.feed_forward(inputs)
        return inputs

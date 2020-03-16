import numpy as np

import activation_functions
from layers import Layer
from loss_functions import *


class NeuralNetwork:
    def __init__(self):
        self.weights = []
        self.network_structure = []

    def _add_weights(self, weights):
        self.weights.append(weights)

    def add_layer(self, n, activation="identity"):
        self.network_structure.append(Layer(n, activation=activation))

    def initialize_weights(self, network_structure):
        self.network_structure = network_structure
        prev_layer_size = network_structure[0]
        for l in range(1, len(network_structure)):
            layer_size = network_structure[l]
            weights_l = np.random.rand(prev_layer_size, layer_size)
            weights_l = weights_l * np.sqrt(2 / prev_layer_size)  # He initializaton
            weights_l = np.vstack((weights_l, np.zeros(layer_size)))  # adding biases
            self._add_weights(weights_l)
            prev_layer_size = layer_size

    def _propagate_one_layer(self, input_values, weights, activation="identity"):
        input_with_one = np.r_[input_values, 1]
        raw_output = np.dot(input_with_one, weights)
        act_functions = {
            "identity": "identity",
            "relu": "ReLU_activation",
            "softmax": "softmax_activation"
        }
        activated_output = getattr(activation_functions, act_functions[activation.lower()])(raw_output)
        return raw_output, activated_output

    def inference(self, input_values, save_outputs=False):
        assert(len(self.weights) is not 0), "No weights in your neural net."
        assert(len(input_values) == (len(self.weights[0]) - 1)), \
            "Input array has incompatible size with input layer."

        raw_outputs, activated_outputs = [], []
        for l in range(len(self.weights) - 1):
            raw_output, activated_output \
                = self._propagate_one_layer(input_values, self.weights[l], activation="relu")
            if save_outputs:
                raw_outputs.append(raw_output)
                activated_outputs.append(activated_output)
            input_values = activated_output

        input_values = np.r_[input_values, 1]
        raw_output = np.dot(input_values, self.weights[-1])
        return softmax_activation(raw_output)

    def train(self, batch):
        assert(len(self.weights) is not 0), "No weights in your neural net."
        assert((len(batch[0]) - 1) == (len(self.weights[0]) - 1)), \
            "Input values have incompatible size with input layer."


def main():
    neural_net = NeuralNetwork()
    neural_net.initialize_network((5, 4, 3))
    print(neural_net.inference(np.array([1, 5, 4, 6, -2])))


if __name__ == '__main__':
    main()

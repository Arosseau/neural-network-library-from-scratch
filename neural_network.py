import numpy as np

import activation_functions
import initializers
import loss_functions

from layers import *


class NeuralNetwork:
    def __init__(self):
        self.weights = []
        self.layers = []

    def _add_weights(self, weights):
        self.weights.append(weights)

    def add(self, layer: Layer):
        self.layers.append(layer)

    def initialize_weights(self, initializer="random"):

        weight_initializer = getattr(initializers, initializers.initializer_fs[initializer.lower()])

        prev_layer_size = self.layers[0].n
        for l in range(1, len(self.layers)):
            layer_size = self.layers[l].n
            weights_l = weight_initializer(prev_layer_size, layer_size)
            weights_l = np.vstack((weights_l, np.zeros(layer_size)))  # adding biases of 0
            self._add_weights(weights_l)
            prev_layer_size = layer_size

    def _propagate_one_layer(self, input_values, weights, activation="identity"):
        m, n = input_values.shape
        input_with_ones = np.ones((m, n + 1))
        input_with_ones[:, :-1] = input_values  # faster than stacking
        raw_output = np.matmul(input_with_ones, weights)

        activated_output = getattr(activation_functions, activation_functions.activation_fs[activation.lower()])(raw_output)
        return raw_output, activated_output

    def inference(self, input_values, save_outputs=False):
        if input_values.ndim == 1:
            input_values = input_values.reshape((1, len(input_values)))

        assert(len(self.weights) is not 0), "No weights in your neural net."
        assert(input_values.shape[1] == (self.weights[0].shape[0] - 1)), \
            "Input array has incompatible size with input layer."

        raw_outputs, activations, activated_outputs = [input_values], ["identity"], [input_values]
        for l in range(len(self.weights)):
            activation_f = self.layers[l+1].activation
            raw_output, activated_output = \
                self._propagate_one_layer(input_values, self.weights[l], activation=activation_f)
            if save_outputs:
                raw_outputs.append(raw_output)
                activations.append(activation_f)
                activated_outputs.append(activated_output)
            input_values = activated_output

        if save_outputs:
            return raw_outputs, activations, activated_outputs
        else:
            return activated_output

    def train(self, batch, loss="quadratic", learning_rate=0.01):
        input_values, labels = batch[:, :-1], batch[:, -1]
        labels = np.eye(max(labels) + 1)[labels]  # one-hot-encoding of numerical labels
        raw_outputs, activations, activated_outputs = self.inference(input_values, save_outputs=True)

        ''' Get loss function and its derivatives:
         ("dx_y" means partial derivative of y to x) '''
        loss = getattr(loss_functions, loss_functions.loss_fs[loss.lower()])(activated_outputs[-1])
        print("Loss: ", loss)
        try:
            da_loss = getattr(loss_functions, "da_" + loss_functions.loss_fs[loss.lower()])(activated_outputs[-1], labels)
            dz_a = getattr(activation_functions, "dz_" + activation_functions.activation_fs[activations[-1]])(raw_outputs[-1])
            dz_loss = np.multiply(da_loss, dz_a)  # Hadamard product
        except AttributeError as e:
            dz_loss = getattr(loss_functions, "dz_" + loss_functions.loss_fs[loss.lower()])(activated_outputs[-1], labels)

        for l in range(1, len(self.weights) + 1):
            m, n = activated_outputs[-l-1].shape
            activated_outputs_with_ones = np.ones((m, n + 1))
            activated_outputs_with_ones[:, :-1] = activated_outputs[-l-1]  # faster than stacking
            dw_loss = np.matmul(activated_outputs_with_ones.T, dz_loss)
            self.weights[-l] = self.weights[-l] - learning_rate * dw_loss / len(batch)

            dz_a = getattr(activation_functions, "dz_" + activation_functions.activation_fs[activations[-l-1]])(raw_outputs[-l-1])
            dz_loss = np.multiply(np.matmul(self.weights[-l][:-1, :], dz_loss), dz_a)  # removed biases


def main():
    neural_net = NeuralNetwork()
    neural_net.add(InputLayer(5))
    neural_net.add(DenseLayer(4, activation="relu"))
    neural_net.add(DenseLayer(3, activation="softmax"))
    neural_net.initialize_weights(initializer="He")

    activated_output, raw_outputs, activated_outputs = \
        neural_net.inference(np.array([[1, -5, -4, 6, -2],
                                      [1, -2, 6.56, -9.56, 7.3]]), save_outputs=True)

    print(activated_output)
    print("raw outputs: ", raw_outputs)
    print("activated outputs: ", activated_outputs)


if __name__ == '__main__':
    main()

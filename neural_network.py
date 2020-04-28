import numpy as np

from utils import get_activation, get_initializer, get_loss
from layers import Layer


class NeuralNetwork:
    def __init__(self):
        self.weights = []
        self.layers = []

    def _add_weights(self, weights):
        self.weights.append(weights)

    def add(self, layer: Layer):
        self.layers.append(layer)

    def initialize_weights(self, initializer="random"):
        weight_initializer = get_initializer(initializer)

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

        activated_output = get_activation(activation)(raw_output)
        return raw_output, activated_output

    def inference(self, input_values, save_outputs=False):
        if input_values.ndim == 1:
            input_values = input_values.reshape((1, len(input_values)))

        assert(len(self.weights) is not 0), "No weights in your neural net."
        assert(input_values.shape[1] == (self.weights[0].shape[0] - 1)), \
            "Input array has incompatible size with input layer."

        raw_outputs, activated_outputs = [input_values], [input_values]
        activations = ["identity"]
        for l in range(len(self.weights)):
            activation_f = self.layers[l+1].activation
            raw_output, activated_output = self._propagate_one_layer(
                input_values, self.weights[l], activation=activation_f
            )
            if save_outputs:
                raw_outputs.append(raw_output)
                activations.append(activation_f)
                activated_outputs.append(activated_output)
            input_values = activated_output

        if save_outputs:
            return raw_outputs, activations, activated_outputs
        else:
            return activated_output

    def train(
            self, batch, labels=None, loss="quadratic", learning_rate=0.01,
            epochs=1, mini_batch_size=1
    ):
        if labels is not None:
            batch = np.c_[batch, labels]

        amount_of_labels = len(set(batch[:, -1]))
        for epoch in range(epochs):
            print("Epoch: ", epoch, end=", ")
            np.random.shuffle(batch)  # avoids correlated mini batches or memorization of order
            avg_loss_epoch = []  # average loss over all samples in batch for this epoch
            sample_i = 0
            while sample_i < (len(batch) - mini_batch_size):
                mini_batch = batch[sample_i:sample_i + mini_batch_size]
                input_values, labels = mini_batch[:, :-1], mini_batch[:, -1]
                # one-hot-encoding of numerical labels:
                labels = np.eye(amount_of_labels)[labels.astype(int)]
                raw_outputs, activations, activated_outputs = self.inference(
                    input_values, save_outputs=True
                )

                ''' Get loss function and its derivatives:
                    ("dx_y" means partial derivative of y to x) '''
                minibatch_loss = get_loss(loss)(activated_outputs[-1], labels)
                avg_loss_epoch.append(minibatch_loss)
                try:
                    da_loss = get_loss(loss, d="da_")(activated_outputs[-1], labels)
                    dz_a = get_activation(activations[-1], d="dz_")(raw_outputs[-1])
                    dz_loss = np.multiply(da_loss, dz_a)  # Hadamard product
                except AttributeError as e:
                    dz_loss = get_loss(loss, d="dz_")(activated_outputs[-1], labels)

                for l in range(1, len(self.weights)):
                    m, n = activated_outputs[-l-1].shape
                    # faster than stacking ones to our activated outputs:
                    activated_outputs_with_ones = np.ones((m, n + 1))
                    activated_outputs_with_ones[:, :-1] = activated_outputs[-l-1]
                    dw_loss = np.matmul(activated_outputs_with_ones.T, dz_loss)
                    self.weights[-l] = self.weights[-l] - learning_rate * dw_loss / len(batch)

                    dz_a = get_activation(activations[-l-1], d="dz_")(raw_outputs[-l-1])
                    dz_loss = np.multiply(
                        np.matmul(dz_loss, self.weights[-l][:-1, :].T),  # removed biases
                        dz_a
                    )

                m, n = activated_outputs[0].shape
                activated_outputs_with_ones = np.ones((m, n + 1))
                activated_outputs_with_ones[:, :-1] = activated_outputs[0]
                dw_loss = np.matmul(activated_outputs_with_ones.T, dz_loss)
                self.weights[0] = self.weights[0] - learning_rate * dw_loss / len(batch)

                sample_i += mini_batch_size

            avg_loss_epoch = np.sum(np.array(avg_loss_epoch)) / np.array(avg_loss_epoch).size
            print("Loss: ", avg_loss_epoch)

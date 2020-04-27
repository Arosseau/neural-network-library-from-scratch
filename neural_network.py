import numpy as np

import activation_functions
import initializers
import loss_functions

from layers import *


def get_function(name: str, type: str, deriv=""):
    if type == "initializer":
        return getattr(
            initializers,
            initializers.initializer_fs[name.lower()]
        )
    if type == "activation":
        return getattr(
            activation_functions,
            deriv + activation_functions.activation_fs[name.lower()]
        )
    if type == "loss":
        return getattr(
            loss_functions,
            deriv + loss_functions.loss_fs[name.lower()]
        )


class NeuralNetwork:
    def __init__(self):
        self.weights = []
        self.layers = []

    def _add_weights(self, weights):
        self.weights.append(weights)

    def add(self, layer: Layer):
        self.layers.append(layer)

    def initialize_weights(self, initializer="random"):

        weight_initializer = get_function(initializer, type="initializer")

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

        activated_output = get_function(activation, type="activation")(raw_output)
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

    def train(self, batch, labels=None, loss="quadratic", learning_rate=0.01, epochs=1, mini_batch_size=1):
        if labels is not None:
            batch = np.c_[batch, labels]

        amount_of_labels = len(set(batch[:, -1]))
        for epoch in range(epochs):
            print("Epoch: ", epoch)
            np.random.shuffle(batch)  # avoids correlated mini batches or memorization of order
            avg_loss_epoch = []  # average loss over all samples in batch for this epoch
            sample_i = 0
            while sample_i < (len(batch) - mini_batch_size):
                mini_batch = batch[sample_i:sample_i + mini_batch_size]
                input_values, labels = mini_batch[:, :-1], mini_batch[:, -1]
                labels = np.eye(amount_of_labels)[labels.astype(int)]  # one-hot-encoding of numerical labels
                raw_outputs, activations, activated_outputs = \
                    self.inference(input_values, save_outputs=True)

                ''' Get loss function and its derivatives:
                 ("dx_y" means partial derivative of y to x) '''
                loss_f = get_function(loss, type="loss")
                minibatch_loss = loss_f(activated_outputs[-1], labels)
                avg_loss_epoch.append(minibatch_loss)
                try:
                    f = get_function(loss, "loss", deriv="da_")
                    da_loss = f(activated_outputs[-1], labels)
                    f = get_function(activations[-1], "activation", deriv="dz_")
                    dz_a = f(raw_outputs[-1])
                    dz_loss = np.multiply(da_loss, dz_a)  # Hadamard product
                except AttributeError as e:
                    f = get_function(loss, "loss", deriv="dz_")
                    dz_loss = f(activated_outputs[-1], labels)

                for l in range(1, len(self.weights)):
                    m, n = activated_outputs[-l-1].shape
                    activated_outputs_with_ones = np.ones((m, n + 1))
                    activated_outputs_with_ones[:, :-1] = activated_outputs[-l-1]  # faster than stacking
                    dw_loss = np.matmul(activated_outputs_with_ones.T, dz_loss)
                    self.weights[-l] = self.weights[-l] - learning_rate * dw_loss / len(batch)

                    f = get_function(activations[-l-1], "activation", deriv="dz_")
                    dz_a = f(raw_outputs[-l-1])
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
            print("loss: ", avg_loss_epoch)


def main():
    def loadMNIST(prefix, folder):
        intType = np.dtype('int32').newbyteorder('>')
        nMetaDataBytes = 4 * intType.itemsize

        data = np.fromfile(folder + "/" + prefix + '-images-idx3-ubyte', dtype='ubyte')
        magicBytes, nImages, width, height = np.frombuffer(data[:nMetaDataBytes].tobytes(), intType)
        data = data[nMetaDataBytes:].astype(dtype='float32').reshape([nImages, width, height])

        labels = np.fromfile(folder + "/" + prefix + '-labels-idx1-ubyte',
                             dtype='ubyte')[2 * intType.itemsize:]

        return data, labels

    train, train_labels = loadMNIST("train", "./mnist/")
    test, test_labels = loadMNIST("t10k", "./mnist/")

    train = train.reshape((len(train), 784)) / 255.
    test = test.reshape((len(test), 784)) / 255.

    # print(train[0])

    neural_net = NeuralNetwork()
    neural_net.add(InputLayer(784))
    neural_net.add(DenseLayer(30, activation="relu"))
    neural_net.add(DenseLayer(10, activation="softmax"))
    neural_net.initialize_weights(initializer="He")

    # print(neural_net.weights[-1][:, 0])

    neural_net.train(train, labels=train_labels.astype(int), loss="cross_entropy", learning_rate=0.1, epochs=10, mini_batch_size=8)

    # raw_outputs, activations, activated_outputs = \
    #     neural_net.inference(np.random.rand(3, 784), save_outputs=True)

    # print(activated_outputs[-1])
    # print("raw outputs: ", raw_outputs)
    # print("activated outputs: ", activated_outputs)


if __name__ == '__main__':
    main()

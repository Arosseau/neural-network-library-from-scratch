import numpy as np


def cross_entropy_loss(prediction, ground_truth):
    return - np.log(np.dot(prediction, ground_truth))


def d_cross_entropy_loss(prediction, ground_truth):
    return - 1. / (prediction * ground_truth)


def ReLU_activation(raw_output):
    return np.maximum(raw_output, np.zeros(len(raw_output)))


def softmax_activation(raw_output):
    exponentials = np.exp(raw_output)
    return exponentials / sum(exponentials)


class NeuralNetwork:
    def __init__(self):
        self.weights = []
        self.layer_structure = None

    def add_weights(self, weights):
        self.weights.append(weights)

    def initialize_network(self, layer_structure):
        self.layer_structure = layer_structure
        prev_layer_size = layer_structure[0]
        for l in range(1, len(layer_structure)):
            layer_size = layer_structure[l]
            weights_l = np.random.rand(prev_layer_size, layer_size)
            weights_l = weights_l * np.sqrt(2 / prev_layer_size)  # He initializaton
            weights_l = np.vstack((weights_l, np.zeros(layer_size)))  # adding biases
            self.add_weights(weights_l)
            prev_layer_size = layer_size

    def inference(self, input_values):
        assert(len(self.weights) is not 0), "No weights in your neural net."
        assert(len(input_values) == (len(self.weights[0]) - 1)), \
            "Input array has incompatible size with input layer."
        for l in range(len(self.weights) - 1):
            input_values = np.r_[input_values, 1]
            raw_output = np.dot(input_values, self.weights[l])
            activated_output = ReLU_activation(raw_output)
            input_values = activated_output

        input_values = np.r_[input_values, 1]
        raw_output = np.dot(input_values, self.weights[-1])
        return softmax_activation(raw_output)

    def gradient_descent(self, prediction, ground_truth):
        assert(len(prediction) == len(ground_truth)), "Ground truth and prediction sizes are not compatible."
        loss = cross_entropy_loss(prediction, ground_truth)
        updated_last_layer_weights = self.weights[-1]


def main():
    neural_net = NeuralNetwork()
    neural_net.initialize_network((5, 4, 3))
    print(neural_net.inference(np.array([1, 5, 4, 6, -2])))


if __name__ == '__main__':
    main()

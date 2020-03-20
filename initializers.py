import numpy as np


initializer_fs = {
            "random": "random_initializer",
            "he": "He_initializer"
        }


def random_initializer(input_layer_size, output_layer_size):
    return np.random.rand(input_layer_size, output_layer_size)


def He_initializer(input_layer_size, output_layer_size):
    weights = np.random.rand(input_layer_size, output_layer_size)
    return weights

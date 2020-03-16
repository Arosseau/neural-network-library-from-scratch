import numpy as np


def identity(raw_output):
    return raw_output


def ReLU_activation(raw_output):
    return np.maximum(raw_output, np.zeros(len(raw_output)))


def softmax_activation(raw_output):
    exponentials = np.exp(raw_output)
    return exponentials / sum(exponentials)

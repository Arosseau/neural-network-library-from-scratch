from abc import ABC, abstractmethod


class Layer(ABC):
    def __init__(self, n):
        self.n = n
        

class InputLayer(Layer):
    def __init__(self, n):
        super().__init__(n)


class DenseLayer(Layer):
    def __init__(self, n, activation="identity"):
        super().__init__(n)
        self.activation = activation

    def set_activation(self, activation):
        self.activation = activation

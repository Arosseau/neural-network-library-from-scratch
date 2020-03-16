class Layer:
    def __init__(self, n, activation="identity"):
        self.n = n
        self.activation = activation

    def set_activation(self, activation):
        self.activation = activation

    def set_number_of_neurons(self, n):
        self.n = n

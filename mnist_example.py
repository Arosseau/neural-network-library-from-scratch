import numpy as np

from layers import *
from neural_network import NeuralNetwork


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

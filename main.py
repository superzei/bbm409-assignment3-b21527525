from nn.neuralnetwork import NeuralNetwork
import numpy as np
import nn.math_ as m

if __name__ == '__main__':
    NN = NeuralNetwork([2, 3, 1], learning_rate=1.0)

    expected = np.random.rand(1)
    inp = np.random.rand(1, 2)
    i = 0
    while True:
        NN.train(inp, expected)
        print("(%f, %f) %d %f %f %f" % (inp[0][0], inp[0][1], i, expected[0], NN.network[-1].output[0], NN.network[-1].error[0]))
        i += 1

    print("finish")

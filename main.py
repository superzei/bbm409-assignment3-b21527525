from nn.neuralnetwork import NeuralNetwork
import numpy as np
from nn.math_ import output2binary, normalize
from data.code.readImage import load_data

"""if __name__ == '__main__':

    train, validation, test = load_data()
    train_x = [normalize(x, 255) for x in train['x']]
    train_y = output2binary(train['y'][0])

    NN = NeuralNetwork([768, 5], learning_rate=0.01)
    # NN.predict([1, 0], [1, 1])
    NN.predict(train_x[0], train_y[0])
    for i in range(1000):
        print(i)
        NN.fit(train_x[:50], train_y[:50], batch_size=1)

        print(NN.last_batch_loss)

        NN.predict(train_x[505], train_y[505])
        print(NN.out_node.vectorize(item='output'))
        print(train_y[505])

        print("-------------------\n")

    print("finish")"""

if __name__ == '__main__':
    train, validation, test = load_data()
    train_x = [normalize(x, 255) for x in train['x']]
    train_y = output2binary(train['y'][0])

    NN = NeuralNetwork([768, 100, 100, 100, 5], learning_rate=0.005)
    # TODO: fix sigmoid returning integer values (1 or 0, not in between)
    for epoch in range(500):
        for i, sample in enumerate(train_x):
            NN.train(sample, train_y[i])

        print(100 - (NN.miss_count / len(train_x)) * 100)
        print(NN.out_layer.output)
        print(sum(NN.out_layer.output))
        print(train_y[-1])
        print("-"*20+"\n")
        NN.miss_count = 0.0

    print("done")

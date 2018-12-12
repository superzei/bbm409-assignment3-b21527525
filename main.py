from nn.neuralnetwork import NeuralNetwork
from nn.math_ import output2binary, normalize
from data.code.readImage import load_data
import matplotlib.pyplot as plt


DEFAULT_DUMP_FILE = 'trained_model.pkl'
MAX_EPOCH = 250
BATCH_SIZE = 5


if __name__ == '__main__':
    # TODO: 1. add bias (optional, shit is not helping)
    # TODO: 2. prevent overfit (add normalization like l1/l2)
    train, validation, test = load_data()
    train_x = [normalize(x, 255) for x in train['x']]
    train_y = output2binary(train['y'][0])

    NN = NeuralNetwork([768, 500, 5], learning_rate=[0.0005, 0.02], decay_rate=0.0001)
    losses = []
    accuracies = []

    for epoch in range(MAX_EPOCH):
        # for i, sample in enumerate(train_x):
        #    NN.train(sample, train_y[i])

        NN.fit(train_x, train_y, batch_size=BATCH_SIZE)

        print("Current epoch: %d" % epoch)
        print("Learning rate: %f" % NN.l_rate)
        print("Loss: %f" % (NN.out_layer.cost / len(train_x)))
        print("Accuracy: %f" % ((NN.hit_count / len(train_x)) * 100))
        print("Last Prediction: {}".format(NN.out_layer.predicted))
        print("Expected: {}".format(train_y[-1]))
        print("Hit count: %d" % NN.hit_count)
        print("-"*20+"\n")

        losses.append(NN.out_layer.cost / len(train_x))
        accuracies.append((NN.hit_count / len(train_x)))
        NN.clean()

        # save model to file
        NN.dump(DEFAULT_DUMP_FILE)

    # graph accuracy and loss
    plt.plot([i for i in range(MAX_EPOCH)], losses, label='Loss')
    plt.plot([i for i in range(MAX_EPOCH)], accuracies, label='Accuracy')
    plt.legend()
    plt.show()

    print("done")


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

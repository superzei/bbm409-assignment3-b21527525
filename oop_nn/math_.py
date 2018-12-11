from math import e, log10
import numpy as np


def sigmoid(x, derive=False):
    if derive:
        return x * (1 - x)
    return 1 / (1 + (e ** (-1 * x)))


def softmax(x, derive=False):
    """ softmax for given output vector x """
    if derive:
        sf = softmax(x)
        return np.multiply(softmax(sf), np.subtract(1, sf))
    y = [(e ** i) for i in x]
    s = sum(y)
    return [i/s for i in y]


def sse(x, derive=False):
    """ error of sum of squares """
    pass


def cross_entropy(batch_expected, batch, derive=False):
    """ cross entropy loss """
    if derive:
        # derivation of loss used for backpropagation and only done for single input/output pair instead of whole batch
        # do not give 2d+ matrices
        return __ce(np.array(batch), np.array(batch_expected), derive=True)

    total = 0.0
    for sample in range(len(batch)):
        total += __ce(batch[sample], batch_expected[sample])
    return total


def __ce(y_prime, y, derive=False):
    if derive:
        # return (-y / y_prime) + ((1 - y)/(1 - y_prime))
        return -(y * (1 - y_prime))

    # return -np.sum(np.dot(y, np.log(y_prime)) + np.multiply(np.subtract(1, y), np.log(np.subtract(1, y_prime))))
    return -np.sum(np.dot(y, np.log(y_prime)))


def output2binary(arr):
    label_count = max(arr) + 1
    return [number2binary(number, label_count) for number in arr]


def number2binary(number, label_count):
    out = [0] * label_count
    out[number] = 1
    return out


def normalize(arr, top):
    return np.divide(arr, top)


def _batch(arr, n):
    return [arr[i:min(i + n, len(arr))] for i in range(0, len(arr), n)]

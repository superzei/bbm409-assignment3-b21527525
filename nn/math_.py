import numpy as np
from math import e
from decimal import Decimal
np.seterr(all='raise')


def sigmoid(x, derivative=False):
    """ logistic function for matrix x """
    if derivative:
        s = sigmoid(x)
        return np.multiply(s, 1 - s)
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x, derivative=False):
    """ softmax for given output vector x """
    if derivative:
        sf = softmax(x)
        # return np.multiply(sf, np.subtract(1, sf))
        s = sf.reshape(-1, 1)
        return np.diagflat(sf) - np.dot(s, s.T)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def cross_entropy(batch, expected, derivative=False):
    """ cross entropy loss """
    if derivative:
        # derivation of loss used for backpropagation and only done for single input/output pair instead of whole batch
        # do not give 2d+ matrices
        return -(np.array(expected) / np.array(batch))
        # return -np.multiply(expected, np.subtract(1, batch))

    # total = -np.sum(np.dot(expected, np.log(batch)) + np.multiply(np.subtract(1, expected),
    #  np.log(np.subtract(1, batch))))
    total = -(np.sum(np.dot(expected, np.log(batch))))
    return total


def signum(x):
    """ step function """
    if x == 0:
        return 0.0
    return 1.0 if x > 0 else -1.0


def square_loss(x, y, derivative=False):
    """ calculate sse for matrix x for given expected matrix y """
    if derivative:
        return x - y
    return np.multiply(np.square(np.array(y, dtype='float64') - np.array(x, dtype='float64')), 0.5)


def relu(x):
    """ rectified linear unit """
    return np.maximum(x, 0.0)


def d_relu(x):
    """ derivative of relu """
    y = x.copy()
    y[y <= 0] = 0.0
    y[y > 0] = 1.0
    return y


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


def get_max_index(x):
    return list(x).index(max(x))

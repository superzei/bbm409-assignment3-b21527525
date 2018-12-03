import numpy as np
from decimal import Decimal


def sigmoid(x):
    """ logistic function for matrix x """
    no_flow = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-no_flow))


def d_sigmoid(x):
    """ derived sigmoid for matrix x """
    return np.multiply(x, 1-x)


def step(x):
    """ step function """
    if x == 0:
        return 0.0
    return 1.0 if x > 0 else -1.0


def square_loss(x, y):
    """ calculate sse for matrix x for given expected matrix y """
    return np.multiply(np.square(np.array(y, dtype='float64') - np.array(x, dtype='float64')), 0.5)


def d_square_loss(x, y):
    """ derived loss for matrix x for given expected matrix y """
    return x - y


def relu(x):
    """ rectified linear unit """
    return np.maximum(x, 0.0)


def d_relu(x):
    """ derivative of relu """
    y = x.copy()
    y[y <= 0] = 0.0
    y[y > 0] = 1.0
    return y

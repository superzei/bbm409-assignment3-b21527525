import numpy as np

# stop on overflow and underflow
np.seterr(all='raise')


def select_activation(string):
    """
    Select activation function
    :param string: name of the activation
    :return: method of activation
    """
    string = string.lower()

    if string == 'sigmoid':
        return sigmoid
    elif string == 'softmax':
        return softmax
    elif string == '':
        return None
    else:
        raise ValueError('{} activation function does not exists in math_ file.'.format(string))


def select_loss(string):
    """
    select a loss function from string
    :param string: name of the loss function
    :return: loss function
    """
    string = string.lower()

    if string == 'cross_entropy':
        return cross_entropy
    else:
        raise ValueError('{} loss function does not exists in math_ file. Check your input.'.format(string))


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
        return -np.array(expected) / np.array(batch)
        # return -np.multiply(expected, np.subtract(1, batch))

    # total = -np.sum(np.dot(expected, np.log(batch)) + np.multiply(np.subtract(1, expected),
    #  np.log(np.subtract(1, batch))))
    preds = np.clip(batch, 1e-12, 1 - 1e-12)
    total = -(np.sum(expected * np.log(preds+1e-9))) / preds.shape[0]
    return total


def delta_softmax_cross_entropy(x, y):
    """
    combined derivative for softmax and cross entropy
    gives derivative of loss w.r.t. output
    """
    return np.subtract(x, y)


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


def unison_shuffle(x, y):
    """ shuffles two lists in unison """
    c = np.c_[x.reshape(len(x), -1), y.reshape(len(y), -1)]
    x_s = c[:, :x.size // len(x)].reshape(x.shape)
    y_s = c[:, x.size // len(x):].reshape(y.shape)
    np.random.shuffle(c)
    return x_s, y_s

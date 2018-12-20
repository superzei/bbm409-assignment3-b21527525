import nn.math_ as m
import numpy as np
np.random.seed(100)


class Layer:
    """ a neural network layer object """
    def __init__(self, layer_info, l_rate):
        # details of layer
        self.layer_type = "hidden"
        self.node_count = layer_info["node_count"]
        self.learning_rate = l_rate

        # count of weights = count of previous layers nodes
        self.weights = np.array([])

        # biases
        self.bias = np.array([])

        # raw values of nodes
        self.input = np.array([])

        # values after activation function
        self.output = np.array([])

        # calculated error
        self.previous_delta = np.array([])
        self.delta = np.zeros((1, self.node_count))

        self.previous_layer = None
        self.next_layer = None

        # function used to activate nodes in this layer
        self.activation_function = m.select_activation(layer_info["activation"])

    def attach(self, previous_layer, next_layer):
        """ attach layer other layers to create a network """
        self.previous_layer = previous_layer
        self.next_layer = next_layer

        # initialize weights
        self.weights = np.random.randn(next_layer.node_count, self.node_count)

        # initialize bias
        self.bias = np.random.randn(next_layer.node_count)

    def forward(self, dropout_probability=0.0):
        self.input = np.dot(self.previous_layer.weights, self.previous_layer.output) + self.previous_layer.bias
        self.output = self.dropout(self.activation_function(self.input), drop_probability=dropout_probability)
        self.next_layer.forward(dropout_probability=dropout_probability)

    def calculate_delta(self):
        error = np.dot(self.next_layer.delta, self.weights) * self.activation_function(self.input, derivative=True)
        self.delta = error

        self.previous_layer.calculate_delta()

    def update(self, momentum_parameter=1.0):
        """ update the weights and bias """
        update = (self.next_layer.delta * self.output.reshape(-1, 1) * self.learning_rate).T

        # initialize the momentum
        if self.next_layer.previous_delta.size == 0:
            self.next_layer.previous_delta = np.zeros(update.shape)

        self.weights -= (update + ((1 - momentum_parameter) * self.next_layer.previous_delta))
        self.bias -= (self.next_layer.delta * self.learning_rate)

        # update the momentum
        self.next_layer.previous_delta = update

        self.next_layer.update(momentum_parameter=momentum_parameter)

    def clean(self):
        """ clean temps, just in case """
        self.output = np.array([])
        self.input = np.array([])
        self.delta = np.zeros((1, self.node_count))

    @staticmethod
    def dropout(x, drop_probability=0.5):
        """ dropout regularization """
        keep_probability = 1 - drop_probability
        mask = np.random.uniform(0.0, 1.0, size=x.shape) < keep_probability

        if keep_probability == 1:
            return x
        elif keep_probability > 0.0:
            scale = (1 / keep_probability)
        else:
            scale = 0.0

        return mask * x * scale


class InputLayer(Layer):
    """ a neural network input layer """
    def __init__(self, layer_info, l_rate):
        super(InputLayer, self).__init__(layer_info, l_rate)
        self.layer_type = "input"

    def forward(self, dropout_probability=0.0):
        self.output = np.array(self.input)
        self.next_layer.forward(dropout_probability=dropout_probability)

    def calculate_delta(self):
        """ stop after reaching input """
        pass


class OutputLayer(Layer):
    """ a neural network output layer """
    def __init__(self, layer_info, l_rate, loss=m.cross_entropy):
        super(OutputLayer, self).__init__(layer_info, l_rate)
        self.layer_type = "output"

        # expected output
        self.expected = []

        # predicted
        self.predicted = []

        # calculated loss for every node on output layer
        self.cost = 0.0

        # functions for error calculation
        self.loss_func = loss

    def attach(self, previous_layer, next_layer):
        self.previous_layer = previous_layer

    def loss(self, expected):
        """ calculate cost for given expected outputs """
        self.expected = expected
        self.cost += self.loss_func(self.predicted, self.expected)

    def calculate_delta(self):
        if self.activation_function == m.softmax:
            self.delta = np.dot(self.loss_func(self.output, self.expected, derivative=True),
                                self.activation_function(self.input, derivative=True))
        else:
            self.delta = np.multiply(self.loss_func(self.output, self.expected, derivative=True),
                                     self.activation_function(self.input, derivative=True))

        # self.delta = m.delta_softmax_cross_entropy(self.predicted, self.expected)

        self.previous_layer.calculate_delta()

    def forward(self, dropout_probability=0.0):
        self.input = np.dot(self.previous_layer.weights, self.previous_layer.output) + self.previous_layer.bias
        self.output = self.activation_function(self.input)
        self.predicted = self.output

    def update(self, momentum_parameter=1.0):
        """ stop after reaching output layer """
        pass

    def clean(self):
        """ squeaky clean """
        super().clean()
        self.cost = 0.0

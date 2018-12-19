import nn.math_ as m
import numpy as np
from math import log
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

    def forward(self):
        self.input = np.dot(self.previous_layer.weights, self.previous_layer.output) + self.previous_layer.bias
        self.output = self.activation_function(self.input)
        self.next_layer.forward()

    def calculate_delta(self):
        error = np.dot(self.next_layer.delta, self.weights) * self.activation_function(self.input, derivative=True)
        self.delta = error

        self.previous_layer.calculate_delta()

    def update(self):
        update = (self.next_layer.delta * self.output.reshape(-1, 1) * self.learning_rate).T
        self.weights -= update
        self.bias -= (self.next_layer.delta * self.learning_rate)
        self.next_layer.update()

    def clean(self):
        """ clean temps, just in case """
        self.output = np.array([])
        self.input = np.array([])
        self.delta = np.zeros((1, self.node_count))


class InputLayer(Layer):
    """ a neural network input layer """
    def __init__(self, layer_info, l_rate):
        super(InputLayer, self).__init__(layer_info, l_rate)
        self.layer_type = "input"

    def forward(self):
        self.output = np.array(self.input)
        self.next_layer.forward()

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
        # self.delta = np.dot(self.loss_func(self.output, self.expected, derivative=True),
        #                          m.softmax(self.input, derivative=True))
        self.delta = m.delta_softmax_cross_entropy(self.predicted, self.expected)

        self.previous_layer.calculate_delta()

    def forward(self):
        self.input = np.dot(self.previous_layer.weights, self.previous_layer.output) + self.previous_layer.bias
        self.output = self.activation_function(self.input)
        self.predicted = self.output

    def update(self):
        """ stop after reaching output layer """
        pass

    def clean(self):
        """ squeaky clean """
        super().clean()

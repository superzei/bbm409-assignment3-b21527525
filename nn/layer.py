import nn.math_ as m
import numpy as np
np.random.seed(100)


class Layer:
    def __init__(self, node_count, l_rate, activation_function=m.sigmoid):

        # debugging
        # np.seterr(all='raise')

        # details of layer
        self.layer_type = "hidden"
        self.node_count = node_count
        self.learning_rate = l_rate

        # count of weights = count of previous layers nodes
        self.weights = np.array([])

        # raw values of nodes
        self.input = np.array([])

        # values after activation function
        self.output = np.array([])

        # calculated error
        self.error = np.array([])
        self.delta = np.zeros((1, self.node_count))

        self.previous_layer = None
        self.next_layer = None

        # function used to activate nodes in this layer
        self.activation_function = activation_function

    def attach(self, previous_layer, next_layer):
        """ attach layer other layers to create a network """
        self.previous_layer = previous_layer
        self.next_layer = next_layer

        # initialize weights
        self.weights = np.random.randn(next_layer.node_count, self.node_count)

    def forward(self):
        self.input = np.dot(self.previous_layer.weights, self.previous_layer.output)
        self.output = self.activation_function(self.input)
        self.next_layer.forward()

    def calculate_delta(self):
        error = np.dot(self.next_layer.delta, self.weights) * self.output
        error *= self.activation_function(self.input, derivative=True) * self.learning_rate
        self.delta += error

        self.previous_layer.calculate_delta()

    def update(self):
        self.weights -= self.next_layer.delta.T
        self.next_layer.update()

    def clean(self):
        """ clean temps, just in case """
        self.output = np.array([])
        self.input = np.array([])
        self.delta = np.zeros((1, self.node_count))


class InputLayer(Layer):
    def __init__(self, node_count, l_rate):
        super(InputLayer, self).__init__(node_count, l_rate)
        self.layer_type = "input"

    def forward(self):
        self.output = np.array(self.input)
        self.next_layer.forward()

    def calculate_delta(self):
        """ stop after reaching input """
        pass


class OutputLayer(Layer):
    def __init__(self, node_count, l_rate, activation_function=m.relu):
        super(OutputLayer, self).__init__(node_count, l_rate, activation_function=activation_function)
        self.layer_type = "output"

        # expected output
        self.expected = []

        # predicted
        self.predicted = []

        # calculated loss for every node on output layer
        self.cost = 0.0

        # functions for error calculation
        self.loss_func = m.cross_entropy

    def attach(self, previous_layer, next_layer):
        self.previous_layer = previous_layer

    def loss(self, expected):
        """ calculate cost for given expected outputs """
        self.expected = expected
        self.cost += self.loss_func(self.output, self.expected)
        self.calculate_delta()

    def calculate_delta(self):
        # self.delta = np.multiply(self.loss_func(self.predicted, self.expected, derivative=True),
        #                          self.activation_function(self.input, derivative=True))

        # self.delta = np.dot(self.delta, m.softmax(self.output, derivative=True))
        self.delta += np.multiply(np.subtract(self.predicted, self.expected),
                                  self.activation_function(self.input, derivative=True))

        self.previous_layer.calculate_delta()

    def forward(self):
        self.input = np.dot(self.previous_layer.weights, self.previous_layer.output)
        self.output = self.activation_function(self.input)
        self.predicted = m.softmax(self.activation_function(self.input))

    def update(self):
        """ stop after reaching output layer """
        pass

    def clean(self):
        """ squeaky clean """
        super().clean()

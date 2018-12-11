import nn.math_ as m
import numpy as np
np.random.seed(1)


class Layer:
    def __init__(self, node_count, l_rate, activation_function=m.sigmoid):

        # debugging
        # np.seterr(all='raise')

        # count of weights = count of previous layers nodes
        self.weights = np.array([])

        # raw values of nodes
        self.input = np.array([])

        # values after activation function
        self.output = np.array([])

        # calculated error
        self.error = np.array([])
        self.delta = np.array([])

        # details of layer
        self.layer_type = "hidden"
        self.node_count = node_count
        self.learning_rate = l_rate

        self.previous_layer = None
        self.next_layer = None

        # function used to activate nodes in this layer
        self.activation_function = activation_function

    def attach(self, previous_layer, next_layer):
        """ attach layer other layers to create a network """
        self.previous_layer = previous_layer
        self.next_layer = next_layer

        # initialize weights
        self.weights = np.random.rand(next_layer.node_count, self.node_count)

    def forward(self):
        self.input = np.dot(self.previous_layer.weights, self.previous_layer.output)
        self.output = self.activation_function(self.input)
        self.next_layer.forward()

    def calculate_delta(self):
        error = np.dot(self.next_layer.delta, self.weights)
        error = np.dot(error, self.output)
        error *= self.activation_function(self.input, derivative=True)
        self.delta = error

        self.previous_layer.calculate_delta()

    def update(self):
        delta = (np.matrix(self.next_layer.delta).T * np.matrix(self.output) * self.learning_rate).__array__()
        self.weights -= delta
        self.next_layer.update()

    def clean(self):
        """ clean temps, just in case """
        self.output = np.array([])
        self.input = np.array([])
        self.delta = np.array([])


class InputLayer(Layer):
    def __init__(self, node_count, l_rate):
        super(InputLayer, self).__init__(node_count, l_rate)
        self.layer_type = "input"

    def forward(self):
        self.output = np.array(self.input)
        self.next_layer.forward()

    def calculate_delta(self):
        pass


class OutputLayer(Layer):
    def __init__(self, node_count, l_rate, activation_function=m.relu):
        super(OutputLayer, self).__init__(node_count, l_rate, activation_function=activation_function)
        self.layer_type = "output"

        # expected output
        self.expected = []

        # calculated loss for every node on output layer
        self.cost = []

        # functions for error calculation
        self.loss_func = m.cross_entropy

    def attach(self, previous_layer, next_layer):
        self.previous_layer = previous_layer

    def loss(self, expected):
        """ calculate cost for given expected outputs """
        self.expected = expected
        self.cost = self.loss_func(self.output, self.expected)
        self.calculate_delta()

    def calculate_delta(self):
        self.delta = np.multiply(self.loss_func(self.output, self.expected, derivative=True),
                                 self.activation_function(self.input, derivative=True))

        self.delta = np.dot(self.delta, m.softmax(self.input, derivative=True))

        self.previous_layer.calculate_delta()

    def forward(self):
        self.input = np.dot(self.previous_layer.weights, self.previous_layer.output)
        self.output = m.softmax(self.activation_function(self.input))

    def update(self):
        """ stop after reaching output layer """
        pass

    def clean(self):
        """ squeaky clean """
        super().clean()
        self.cost = np.array([])
        self.delta = np.array([])

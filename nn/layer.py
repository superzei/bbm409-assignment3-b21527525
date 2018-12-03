import nn.math_ as m
import numpy as np


class Layer:
    def __init__(self, node_count, l_rate, activation_function=m.sigmoid, derived_activation_function=m.d_sigmoid):

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

        # details of layer
        self.layer_type = "hidden"
        self.node_count = node_count
        self.learning_rate = l_rate

        self.previous_layer = None
        self.next_layer = None

        # function used to activate nodes in this layer
        self.activation_function = activation_function
        self.d_activation_function = derived_activation_function

    def attach(self, previous_layer, next_layer):
        """ attach layer other layers to create a network """
        self.previous_layer = previous_layer
        self.next_layer = next_layer

        self.weights = np.random.rand(next_layer.node_count, self.node_count)

    def forward(self):
        self.input = np.dot(self.previous_layer.output, np.transpose(self.previous_layer.weights))
        self.output = self.activation_function(self.input)

        if self.layer_type != "output":
            self.next_layer.forward()

    def delta(self):
        """ calculate error for backpropagation """
        error = np.dot(np.transpose(self.weights), self.next_layer.error)
        derived_activation = self.d_activation_function(self.output)

        # TODO: fix this "drop to 0" bullshit
        self.error = np.multiply(error, np.transpose(derived_activation))

        if self.layer_type != "input":
            self.previous_layer.delta()

    def update(self):
        """ iterate forwards while updating weights using calculated deltas """
        delta = np.dot(self.next_layer.error, self.output)
        delta = np.multiply(self.learning_rate, delta)
        self.weights -= delta
        self.next_layer.update()


class InputLayer(Layer):
    def __init__(self, node_count, l_rate, activation_function=m.relu, derived_activation_function=m.d_relu):
        super(InputLayer, self).__init__(node_count, l_rate, activation_function=activation_function,
                                         derived_activation_function=derived_activation_function)
        self.layer_type = "input"

    def forward(self):
        self.output = np.array(self.input)
        self.next_layer.forward()


class OutputLayer(Layer):
    def __init__(self, node_count, l_rate, activation_function=m.relu, derived_activation_function=m.d_relu):
        super(OutputLayer, self).__init__(node_count, l_rate, activation_function=activation_function,
                                          derived_activation_function=derived_activation_function)
        self.layer_type = "output"

        # expected output
        self.expected = []

        # calculated loss for every node on output layer
        self.cost = []

        # functions for error calculation
        self.loss_func = m.square_loss
        self.d_loss_func = m.d_square_loss

    def attach(self, previous_layer, next_layer):
        self.previous_layer = previous_layer

    def loss(self, expected):
        """ calculate cost for given expected outputs """
        self.expected = expected
        self.cost = self.loss_func(self.output, self.expected)
        self.delta()

    def delta(self):
        """ calculate derived errors for backpropagation """
        self.error = np.dot(self.d_loss_func(self.output, self.expected), self.d_activation_function(self.input))
        self.previous_layer.delta()

    def update(self):
        """ stop after reaching output layer """
        pass

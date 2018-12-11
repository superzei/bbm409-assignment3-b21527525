from oop_nn.node import Node
from oop_nn.math_ import softmax, cross_entropy


class Layer:

    def __init__(self, node_count, learning_rate, layer_type="hidden", activation=lambda x: x):
        self.node_count = node_count
        self.layer_type = layer_type
        self.nodes = []
        self.learning_rate = learning_rate

        self.previous_layer = None
        self.next_layer = None

        self.activation = activation
        self.loss = cross_entropy

        self.error = 0.0

        self.init_layer()

    def init_layer(self):
        self.nodes = [Node(self) for _ in range(self.node_count)]

    def connect(self, previous_layer, next_layer):
        self.previous_layer = previous_layer
        self.next_layer = next_layer

        for index, node in enumerate(self.nodes):
            node.connect_node()
            node.index = index

    def map_input(self, data):
        if len(data) != len(self.nodes):
            raise IndexError("Cannot map given data as input. Length {} and {} does not match."
                             .format(len(data), len(self.nodes)))

        for index, node in enumerate(self.nodes):
            node.input = data[index]

    def vectorize(self, item='output'):
        if item == 'output':
            return [node.output for node in self.nodes]
        elif item == 'input':
            return [node.input for node in self.nodes]
        elif item == 'delta':
            return [node.delta for node in self.nodes]
        elif item == 'weight':
            return [node.weights for node in self.nodes]
        else:
            raise ValueError("Invalid item")

    def forward(self):
        """ forward propagation """
        if self.activation == softmax:
            # softmax requires all the data to be vectorized so it is calculated differently
            activated = softmax(self.vectorize(item='input'))
            for index, node in enumerate(self.nodes):
                node.output = activated[index]
            return

        for node in self.nodes:
            node.fpass()

    def backward(self, delta):
        for index, node in enumerate(self.nodes):
            if self.layer_type == 'output':
                node.delta = delta[index]
            node.bprop()


from random import random, seed
from oop_nn.math_ import softmax
import numpy as np
seed(1)


class Node:

    def __init__(self, layer):
        self.layer = layer
        self.delta = 0.0

        self.forward_node_count = 0
        self.weights = []

        self.forward = []
        self.backward = []

        self.input = 0.0
        self.output = 0.0

        self.index = 0

    def connect_node(self):
        if self.layer.next_layer is not None:
            self.forward_node_count = self.layer.next_layer.node_count
            self.weights = [random() for _ in range(self.forward_node_count)]
            self.forward = self.layer.next_layer.nodes

        if self.layer.previous_layer is not None:
            self.backward = self.layer.previous_layer.nodes

    def fpass(self):
        self.output = self.layer.activation(self.input)

        for index, node in enumerate(self.forward):
            node.input += self.output * self.weights[index]

    def bprop(self):
        if self.layer.activation == softmax:
            deactivated = softmax(self.layer.vectorize(item='input'), derive=True)[self.index]
        else:
            deactivated = self.layer.activation(self.input, derive=True)

        if self.layer.layer_type != 'output':
            errors = self.layer.next_layer.vectorize(item='delta')
            self.delta = sum(np.multiply(errors, self.weights)) * deactivated * self.layer.learning_rate * self.input
            self.weights = list(np.subtract(self.weights, self.delta))
        else:
            self.delta *= deactivated * self.layer.learning_rate




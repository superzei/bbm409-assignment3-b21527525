from oop_nn.layer import Layer
from oop_nn.math_ import sigmoid, softmax, cross_entropy, _batch


class NeuralNetwork:

    def __init__(self, scheme, learning_rate=0.01):
        self.scheme = scheme
        self.rate = learning_rate

        self.data = []
        self.expected = []
        self.batch_error = []
        self.last_batch_loss = 0.0

        self.network = []

        self.in_node = None
        self.out_node = None

        self.build_layers(self.scheme)
        self.build_network()

    def build_layers(self, scheme):
        self.network.append(Layer(scheme[0], self.rate, layer_type='input', activation=sigmoid))

        for nodes in scheme[1:-1]:
            self.network.append(Layer(nodes, self.rate, activation=sigmoid))

        self.network.append(Layer(scheme[-1], self.rate, layer_type='output', activation=softmax))

        self.in_node = self.network[0]
        self.out_node = self.network[-1]

    def build_network(self):
        self.in_node.connect(None, self.network[1])
        self.out_node.connect(self.network[-2], None)

        for layer in range(1, len(self.network) - 1):
            self.network[layer].connect(self.network[layer-1], self.network[layer+1])

    def clean_up(self):
        for layer in self.network:
            layer.map_input([0.0] * layer.node_count)

    def predict(self, test_data, expected_result):
        self.clean_up()
        self.in_node.map_input(test_data)

        for layer in self.network:
            layer.forward()

        # self.loss(expected_result)
        return self.out_node.vectorize()

    def train(self, error_rate, batch_index):
        delta = cross_entropy(self.expected[batch_index][-1], self.out_node.vectorize(), derive=True) * error_rate
        for layer in self.network[::-1]:
            layer.backward(delta)

        # print(self.in_node.vectorize(item='weight'))

        """print(self.expected[batch_index][-1])
        print(self.out_node.vectorize(item='output'))
        print("---------------------------\n")"""
        # print(layer.vectorize(item='delta'))

    def fit(self, x, y, batch_size=100):
        self.data = _batch(x, batch_size)
        self.expected = _batch(y, batch_size)
        for batch_index, batch in enumerate(self.data):
            batch_error = []
            for sample_index, sample in enumerate(batch):
                batch_error.append(
                    self.predict(sample, self.expected[batch_index][sample_index])
                )
            batch_loss = cross_entropy(self.expected[batch_index], batch_error)
            batch_loss /= batch_size
            self.last_batch_loss = batch_loss

            # put backpropagation call here
            self.train(batch_loss, batch_index)

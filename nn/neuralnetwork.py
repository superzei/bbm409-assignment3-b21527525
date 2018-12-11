from nn.layer import Layer, InputLayer, OutputLayer
import nn.math_ as m


class NeuralNetwork:    
    def __init__(self, shape, learning_rate=1.0):
        self.l_rate = learning_rate

        # create input and output layers
        input_layer = InputLayer(shape[0], learning_rate)
        output_layer = OutputLayer(shape[-1], learning_rate, activation_function=m.sigmoid)

        # predictions
        self.predicts = []
        self.miss_count = 0.0

        # create hidden layers
        self.network = [input_layer]
        for layer in range(1, len(shape)-1):
            self.network.append(Layer(shape[layer], learning_rate,  activation_function=m.sigmoid))
        self.network.append(output_layer)

        self.in_layer = self.network[0]
        self.out_layer = self.network[-1]

        # attach input and output
        self.in_layer.attach(None, self.network[1])
        self.out_layer.attach(self.network[-2], None)

        # attach the hidden layers
        for layer in range(1, len(self.network) - 1):
            self.network[layer].attach(self.network[layer - 1], self.network[layer + 1])

    def train(self, sample, output):
        """ train network by given input output pair """
        self.reset()
        self.predict(sample)
        self.out_layer.loss(output)

        if m.get_max_index(self.out_layer.output) != m.get_max_index(output):
            self.miss_count += 1

        self.in_layer.update()

    def predict(self, sample):
        """ predict an output for given sample """
        self.in_layer.input = sample
        self.in_layer.forward()
        return self.out_layer.output

    def reset(self):
        """ clean temporary variables in the network, jic """
        for layer in self.network:
            layer.clean()

from nn.layer import Layer, InputLayer, OutputLayer
import nn.math_ as m


class NeuralNetwork:    
    def __init__(self, shape, learning_rate=1.0):
        self.l_rate = learning_rate

        # create input and output layers
        input_layer = InputLayer(shape[0], learning_rate, activation_function=m.relu,
                                 derived_activation_function=m.d_relu)
        output_layer = OutputLayer(shape[-1], learning_rate, activation_function=m.relu,
                                   derived_activation_function=m.d_relu)

        # create hidden layers
        self.network = [input_layer]
        for layer in range(1, len(shape)-1):
            self.network.append(Layer(shape[layer], learning_rate,  activation_function=m.relu,
                                      derived_activation_function=m.d_relu))
        self.network.append(output_layer)

        # attach input and output
        self.network[0].attach(None, self.network[1])
        self.network[-1].attach(self.network[-2], None)

        # attach the hidden layers
        for layer in range(1, len(self.network) - 1):
            self.network[layer].attach(self.network[layer - 1], self.network[layer + 1])

    def train(self, sample, output):
        self.network[0].input = sample
        self.network[0].forward()
        self.network[-1].loss(output)
        self.network[0].update()

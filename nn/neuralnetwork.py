from nn.layer import Layer, InputLayer, OutputLayer
import nn.math_ as m


class NeuralNetwork:    
    def __init__(self, shape, learning_rate=(0.001, 0.1), decay_rate=0.0005):
        self.l_rate_bound = learning_rate
        self.l_rate = learning_rate[1]
        self.decay_rate = decay_rate

        # create input and output layers
        input_layer = InputLayer(shape[0], self.l_rate)
        output_layer = OutputLayer(shape[-1], self.l_rate, activation_function=m.sigmoid)

        # predictions
        self.predicts = []
        self.hit_count = 0.0

        # create hidden layers
        self.network = [input_layer]
        for layer in range(1, len(shape)-1):
            self.network.append(Layer(shape[layer], self.l_rate,  activation_function=m.sigmoid))
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
        """ train network by given input output pair / DEPRECIATED FOR BATCH """
        self.predict(sample)
        self.out_layer.loss(output)

        if m.get_max_index(self.out_layer.output) != m.get_max_index(output):
            self.hit_count += 1

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

    def decay(self):
        """ decay the learning rate """
        if self.l_rate > self.l_rate_bound[0] and self.l_rate - self.decay_rate > 0.0:
            self.l_rate -= self.decay_rate
        elif self.l_rate - self.decay_rate <= 0.0 or self.l_rate < self.l_rate_bound[0]:
            self.l_rate = self.l_rate_bound[0]

        for layer in self.network:
            layer.learning_rate = self.l_rate

    def __train_batch(self, x, y):
        self.reset()

        for index, batch in enumerate(x):
            self.predict(batch)
            self.out_layer.loss(y[index])

            if m.get_max_index(self.out_layer.predicted) == m.get_max_index(y[index]):
                self.hit_count += 1

        self.in_layer.update()

    def fit(self, sample, expected, batch_size=20):
        """ batch, batch all day long, batch while I sing this song """
        batched_x = m._batch(sample, batch_size)
        batched_y = m._batch(expected, batch_size)

        for index in range(batch_size):
            self.__train_batch(batched_x[index], batched_y[index])

    def clean(self):
        self.decay()
        self.hit_count = 0.0
        self.out_layer.cost = 0.0

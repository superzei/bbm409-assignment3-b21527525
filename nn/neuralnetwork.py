from nn.layer import Layer, InputLayer, OutputLayer
import nn.math_ as m
import numpy as np
import pickle
import json


class NeuralNetwork:
    def __init__(self, shape, config, dropout_probability=0.0):
        """
        Simple neural network object
        :param shape: dictionary of layers. Contains information required to build network
        :param config: configuration for the neural network
        """
        self.l_rate_bound = config['learning_rate_bounds']
        self.l_rate = self.l_rate_bound[1]
        self.decay_rate = config['decay_rate']
        self.default_dropout_chance = dropout_probability
        self.dropout_probability = self.default_dropout_chance
        self.momentum_parameter = config['momentum_parameter']

        self.epochs = config['epochs']
        self.loss_function = m.select_loss(config['loss'])
        self.batch_size = config['batch_size']

        self.batch_loss = 0.0

        # create input and output layers
        input_layer = InputLayer(shape["input"], self.l_rate)
        output_layer = OutputLayer(shape["output"], self.l_rate, loss=self.loss_function)

        # predictions
        self.predicts = []
        self.hit_count = 0.0

        # create hidden layers
        self.network = [input_layer]
        for layer in range(1, len(shape)-1):
            self.network.append(Layer(shape["hidden_"+str(layer)], self.l_rate))
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
        self.predict(sample, dropout_probability=self.dropout_probability)
        self.out_layer.loss(output)

        if m.get_max_index(self.out_layer.output) != m.get_max_index(output):
            self.hit_count += 1

        self.in_layer.update(momentum_parameter=self.momentum_parameter)

    def predict(self, sample, dropout_probability=0.0):
        """ predict an output for given sample """
        self.in_layer.input = sample
        self.in_layer.forward(dropout_probability=dropout_probability)
        return self.out_layer.output

    def bulk_predict(self, x, y):
        """ predicts all the sample in given array """

        # clear all the temporary data
        self.reset()

        for index, sample in enumerate(x):
            self.predict(sample, dropout_probability=self.dropout_probability)
            self.out_layer.loss(y[index])

            # gimme the hit rate
            if m.get_max_index(self.out_layer.predicted) == m.get_max_index(y[index]):
                self.hit_count += 1.0

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
        """ trains whole batch then calculates loss """
        self.reset()

        for index, batch in enumerate(x):
            self.predict(batch, dropout_probability=self.dropout_probability)
            self.out_layer.loss(y[index])

            # increment hit rate if, well, hit
            if m.get_max_index(self.out_layer.predicted) == m.get_max_index(y[index]):
                self.hit_count += 1.0

        # calculate batch loss
        self.batch_loss += (self.out_layer.cost / len(x))

        # calculate all delta
        self.out_layer.calculate_delta()

        # update weights
        self.in_layer.update(momentum_parameter=self.momentum_parameter)

    def fit(self, sample, expected):
        """ batch, batch all day long, batch while I sing this song """

        # shuffle the data
        shuffled_sample, shuffled_expected = m.unison_shuffle(np.array(sample), np.array(expected))

        # batch up
        batched_x = m.batch_(shuffled_sample, self.batch_size)
        batched_y = m.batch_(shuffled_expected, self.batch_size)

        # learning time
        for index in range(len(batched_x)):
            self.__train_batch(batched_x[index], batched_y[index])

    def clean(self):
        """ clean model for new epoch """
        self.decay()
        self.hit_count = 0.0
        self.out_layer.cost = 0.0
        self.batch_loss = 0.0

    def dump(self, fname):
        """ dump the model to file """
        with open(fname, 'wb') as out:
            pickle.dump(self, out, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(fname):
        """ load model from file using pickle """
        with open(fname, 'rb') as inp:
            model = pickle.load(inp)

        if type(model) != NeuralNetwork:
            raise ImportError('Given file is not a neural network')

        return model

    def validate(self, validation_data):
        """ validate model, depreciated """
        self.clean()
        x = m.normalize(validation_data['x'], 255)
        y = m.output2binary(validation_data['y'][0])
        hit_rate = sum([m.get_max_index(self.predict(x[i])) == m.get_max_index(y[i]) for i in range(len(x))])
        return hit_rate

    @staticmethod
    def create(fpath):
        """ create model from template(json file) """
        model_info = json.load(open(fpath))

        model_shape = model_info['model']
        model_settings = model_info['config']
        dropout_chance = model_info['config']['dropout_chance']

        nn = NeuralNetwork(model_shape, model_settings, dropout_probability=dropout_chance)
        return nn

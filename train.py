from nn.neuralnetwork import NeuralNetwork
from nn.math_ import output2binary, normalize
from data.code.readImage import load_data
import matplotlib.pyplot as plt
import sys
import argparse


DEFAULT_DUMP_FILE = 'trained_model.pkl'  # default model file name
SHOW_PLOT = True  # show graph of change in accuracy and loss over epochs
VALIDATE_DATA = True  # use a validation on training (requires validation path from cli)
DUMP_BEST_MODEL = True  # only dump model if accuracy improved


def parameter_parse(argv):
    """
    parses given commandline arguments and flags
    :param argv: command line argument
    :return: path to data
    """
    parser = argparse.ArgumentParser('Train a neural network')
    parser.add_argument('--data_path', '-d', required=True,
                        dest='data_path', type=str, help='Path to the training data')
    parser.add_argument('--model_name', '-m', type=str, default=DEFAULT_DUMP_FILE,
                        dest='model_name', help='name for model, if not given default name will be used')
    parser.add_argument('--template_path', '-t', required=False, type=str, default='model.json',
                        dest='template_path', help='template from file, if not given default path will be used')
    parser.add_argument('--validation_path', '-v', required=False, type=str, default='data/code/validation.mat'
                        , dest='validation_path', help='path to validation data')

    # TODO: add model training from json file

    return vars(parser.parse_args(argv))


if __name__ == '__main__':

    # parse command line arguments
    args = parameter_parse(sys.argv[1:])

    # prepare training data
    train = load_data(args['data_path'])
    train_x = [normalize(x, 255) for x in train['x']]
    train_y = output2binary(train['y'][0])

    # load validation data
    validate = None
    if VALIDATE_DATA:
        try:
            validate = load_data(args['validation_path'])
        except FileNotFoundError:
            print('Validation file not found on given path, omitting\n')
            VALIDATE_DATA = False

    # build model from template
    NN = NeuralNetwork.create(args['template_path'])

    # log the data for graph
    losses = []
    accuracies = []
    validation_accuracies = []

    for epoch in range(NN.epochs):

        NN.dropout_probability = 0.5
        NN.fit(train_x, train_y)

        # get results
        print("Current epoch: %d" % epoch)
        print("Learning rate: %f" % NN.l_rate)
        print("Loss: %f" % NN.batch_loss)
        print("Accuracy: %f" % ((NN.hit_count / len(train_x)) * 100))
        print("Hit count: %d" % NN.hit_count)

        # accumulate the graph data
        losses.append(NN.batch_loss)
        accuracies.append((NN.hit_count / len(train_x)))

        if VALIDATE_DATA:
            # Use validation data
            NN.dropout_probability = 0.0
            validation_accuracy = (NN.validate(validate) / len(validate['x']))
            print("Validation accuracy: {}".format(validation_accuracy * 100))
            validation_accuracies.append(validation_accuracy)

            if max(validation_accuracies) <= validation_accuracy and DUMP_BEST_MODEL:
                # dump model if accuracy improved
                print("Model dumped")
                NN.dump(args['model_name'])
        else:
            # Validation data is not available, use training accuracy instead

            if max(accuracies) <= ((NN.hit_count / len(train_x)) * 100) and DEFAULT_DUMP_FILE:
                # save model to file if accuracy of it improved
                print("Model dumped")
                NN.dump(args['model_name'])

        if not DUMP_BEST_MODEL:
            # dump model to file at every epoch
            NN.dump(args['model_name'])

        # clean model for new epoch
        NN.clean()

        print("-"*20+"\n")

    if SHOW_PLOT:
        # graph accuracies and loss
        fig, axs = plt.subplots(1, 2)

        if VALIDATE_DATA:
            # also plot validation over epochs if available
            axs[0].plot([i for i in range(NN.epochs)], validation_accuracies, label='Validation accuracy')

        axs[0].plot([i for i in range(NN.epochs)], accuracies, label='Training Accuracy')
        axs[1].plot([i for i in range(NN.epochs)], losses, label='Loss')

        axs[0].set_xlabel("Epochs")
        axs[1].set_xlabel("Epochs")

        axs[0].set_title("Accuracy changes over epochs")
        axs[1].set_title("Loss change over epochs")

        axs[0].legend()
        axs[1].legend()

        plt.show()

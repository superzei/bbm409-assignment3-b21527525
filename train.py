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
VISUALIZE = True  # visualize output layers parameters as image, only for single layer


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
    parser.add_argument('--validation_path', '-v', required=False, type=str, default='data/code/validation.mat',
                        dest='validation_path', help='path to validation data')

    return vars(parser.parse_args(argv))


if __name__ == '__main__':
    # TODO: add pre-trained models in trained models
    # TODO: clean trained models

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

    if len(NN.network) > 2:
        # if not single layer network do not visualize even if flag is set
        VISUALIZE = False

    # log the data for graph
    losses = []
    accuracies = []
    validation_accuracies = []

    for epoch in range(NN.epochs):

        NN.dropout_probability = NN.default_dropout_chance
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
        fig, axs = plt.subplots(4, 2) if VISUALIZE else plt.subplots(1, 2)
        plt0 = axs[0, 0] if VISUALIZE else axs[0]
        plt1 = axs[0, 1] if VISUALIZE else axs[1]

        if VALIDATE_DATA:
            # also plot validation over epochs if available
            plt0.plot([i for i in range(NN.epochs)], validation_accuracies, label='Validation accuracy')

        plt0.plot([i for i in range(NN.epochs)], accuracies, label='Training Accuracy')
        plt1.plot([i for i in range(NN.epochs)], losses, label='Loss')

        plt0.set_xlabel("Epochs")
        plt1.set_xlabel("Epochs")

        plt0.set_title("Accuracy changes over epochs")
        plt1.set_title("Loss change over epochs")

        if VISUALIZE:
            ims = []
            for im in NN.out_layer.previous_layer.weights:
                ims.append(im.reshape(32, 24))

            axs[1, 0].imshow(ims[0], cmap='gray')
            axs[1, 1].imshow(ims[1], cmap='gray')
            axs[2, 0].imshow(ims[2], cmap='gray')
            axs[2, 1].imshow(ims[3], cmap='gray')
            axs[3, 0].imshow(ims[4], cmap='gray')

            fig.delaxes(axs[3, 1])

        plt0.legend()
        plt1.legend()

        plt.show()

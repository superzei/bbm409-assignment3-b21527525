from nn.neuralnetwork import NeuralNetwork
from data.code.readImage import load_data
from nn.math_ import normalize, output2binary
import argparse
import sys


def parameter_parse(argv):
    """
    parses given commandline arguments and flags
    :param argv: command line argument
    :return: path to data
    """
    parser = argparse.ArgumentParser('Predict flower from a given .mat file')
    parser.add_argument('--data_path', '-d', required=True,
                        dest='data_path', type=str, nargs=1, help='Path to the training data')
    parser.add_argument('--model_path', '-m', type=str, required=True, nargs=1,
                        dest='model_path', help='path to trained model')

    return vars(parser.parse_args(argv))


if __name__ == '__main__':

    # parse command line arguments
    args = parameter_parse(sys.argv[1:])

    # load data and model
    nn = NeuralNetwork.load(args['model_path'][0])
    test = load_data(args['data_path'][0])

    # clean the data
    test_x = [normalize(x, 255) for x in test['x']]
    test_y = output2binary(test['y'][0])

    # make prediction
    nn.bulk_predict(test_x, test_y)

    # get results
    print("Accuracy: %f" % ((nn.hit_count / len(test_x)) * 100))
    print("Hit count: %d" % nn.hit_count)

    print('done')

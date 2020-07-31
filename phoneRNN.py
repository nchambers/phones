#
# @author Nate Chambers
#
# Main program for training an LSTM for phone number recognition.
# This uses the bilstm.py model, but manages the command line calls and execution.
# Don't use bilstm.py directly, instead run this.
#
# phoneRNN.py train <full-phone.train>
# phoneRNN.py [test|traintest] <full-phone.train> <full-phone.test>
#

import sys
import os
import argparse

import bilstm as rnn

# For the command-line argument parser.
def restricted_float(x):
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x

# Argument Parsing
parser = argparse.ArgumentParser(description="phoneRNN.py")

# Option parsing
parser.add_argument('-e', action="store", dest="edim", type=int, default=100, help='Set Embedding Dimension, default is 100')
parser.add_argument('-i', action="store",dest="idim", type=int, default=200, help='Set RNN Internal Dimension, default is 200')
parser.add_argument('-d', action="store", dest="drop", type=restricted_float, default=0.5, help='Set Dropout in range 0-1, default is 0.5')
parser.add_argument('-b', action="store", dest="batch", type=int, default=32, help='Set Batch size, default is 50')
parser.add_argument('-att', action="store_true", default=False, help='Use Attention Mode')
parser.add_argument('-satt', action="store_true", default=False, help='Use Shared Attention Mode')
parser.add_argument('-con', action="store_true", default=False, help='Use Conditioned Output')
parser.add_argument('-stack', action="store_true", default=False, help='Use Stack RNN')
parser.add_argument('-pre', action="store_true", default=False, help='Use Pre-Training')
parser.add_argument('-cnn', action="store_true", default=False, help='Use Visual Chars CNN')
parser.add_argument('-charsim', action="store_true", default=False, help='Include visual char sim vectors')
parser.add_argument('-onlycharsim', action="store_true", default=False, help='Only input is visual char sim vectors')
parser.add_argument('-jiggle', action="store_true", default=False, help='Jiggle images during training')
parser.add_argument('-repeat', action="store", dest="repeat", type=int, default=0, help='# times to repeat training data with jiggled char images')
parser.add_argument('-exp', action="store_true", default=False, help='Use Expanded Tokens')
parser.add_argument('-epochs', action="store", dest="epochs", type=int, default=75, help='Number of epochs during training')
parser.add_argument('-gpu', action="store", dest="gpu", type=int, default=0, help='Set GPU id to run on, default is 0')
parser.add_argument('-interactive', action="store_true", default=False, help='Interactive console mode after testing runs')
parser.add_argument('-models', action="store", dest="directory", default="models", help='directory to store the model files')

# Subparsers for required test, train, or traintest
subparsers = parser.add_subparsers(help='One of these is required.', dest='mode')
train_parser = subparsers.add_parser("train")
test_parser = subparsers.add_parser("test")
train_test_parser = subparsers.add_parser("traintest")

# Subparser arguments
train_parser.add_argument('trainFile', action="store", metavar="input-tsv-file", help='Name of file containing training data')
test_parser.add_argument('modelPrefix', action="store", metavar="modelname", help='Name of the model files to load')
test_parser.add_argument('testFile', action="store", metavar="input-tsv-file", help='Name of file containing testing data')
train_test_parser.add_argument('trainFile', action="store", metavar="input-tsv-file", help='Name of file containing training data' )
train_test_parser.add_argument('testFile', action="store", metavar="input-tsv-file", help='Name of file containing testing data')

# Parse
args = parser.parse_args()
if args.mode == None:
    print("phoneRNN train|test|traintest <params>")
    sys.exit(1)

# Setup the RNN settings.
rnn.MAX_EPOCHS = args.epochs
rnn.BATCH_SIZE = args.batch
rnn.EMBEDDING_DIM = args.edim #up by 50
rnn.RNN_INTERNAL_DIM = args.idim #Keep bigger 
rnn.DROPOUT = args.drop #between 0-1
rnn.USE_ATTENTION = args.att 
rnn.USE_SHARED_ATTENTION = args.satt
rnn.CONDITIONED_OUTPUT = args.con
rnn.STACK_RNN = args.stack
rnn.USE_PRETRAINING = args.pre
rnn.USE_VISUAL_CNN = args.cnn
rnn.USE_CHAR_SIMILARITIES = args.charsim
rnn.ONLY_CHAR_SIMILARITIES = args.onlycharsim
rnn.JIGGLE_IMGS = args.jiggle
rnn.REPEAT_DATA = args.repeat
rnn.EXPAND_TOKENS = args.exp
rnn.MODEL_DIRECTORY = args.directory
rnn.printSettings()

INTERACTIVE_MODE = args.interactive

# Check if model directory exists, make if not, also add backslash
if rnn.MODEL_DIRECTORY[-1] != '/':
    rnn.MODEL_DIRECTORY = rnn.MODEL_DIRECTORY + '/'
if not os.path.isdir(rnn.MODEL_DIRECTORY):
    try:
        os.mkdir(rnn.MODEL_DIRECTORY)
    except OSError:
        print("Could not create directory" + rnn.MODEL_DIRECTORY)
    else:
        print("Created new model directory" + rnn.MODEL_DIRECTORY)

GPU = args.gpu
print("GPU ID: ", GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)

print()
print("--------------------------------")

# SETUP file paths for training and/or testing.
try:
    # Test for train file
    args.trainFile
except AttributeError:
    # No train file
    trainFile = False
    modelPrefix = args.modelPrefix
    testFile = args.testFile
    print('Testing file: ' + testFile)
    print('Using model: ' + modelPrefix)
else:
    # Train file exists
    try:
        # Test for test file
        args.testFile
    except AttributeError:
        # No test file
        trainFile = args.trainFile
        testFile = False
        print('Training from file: ' + trainFile)
    else:
        trainFile = args.trainFile
        testFile = args.testFile
        print('Testing file: ' + testFile)
        print('Training from file: ' + trainFile)

# TRAIN
if trainFile:
    modelPrefix = rnn.train(trainFile)

# TEST
if testFile:
    print(modelPrefix)
    rnn.test(modelPrefix, testFile, interactive=INTERACTIVE_MODE)

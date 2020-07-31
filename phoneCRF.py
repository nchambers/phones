#
# @author Nate Chambers
#
# phoneCRF.py traintest full.crf.train full.test
#
# This trains on a CRF-formatted input file, but it tests on a
# normal non-CRF-formatted input file of obscured phone numbers
#

import sys
import os
import argparse
import math
import pandas as pd
import numpy as np
from helper import readUnicodeTSV, saveAll, loadAllCRF, levenshtein_distance
import csv
import pickle as pickle
from keras.backend import shape
from keras.models import Model
from keras.models import model_from_json, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Masking, multiply, Lambda, Concatenate
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras_contrib.layers import CRF
from keras_contrib.metrics import crf_viterbi_accuracy
from keras_contrib.losses import crf_loss
import imagesCNN
import images
import tensorflow as tf

# For the command-line argument parser.
def restricted_float(x):
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x

#Argument parser
parser = argparse.ArgumentParser(description="phoneCRF.py")

# Option parsing
parser.add_argument('-ml', action="store", dest="maxlen", type=int, default=75, help='Set Maximum Length, default is 75')
parser.add_argument('-e', action="store", dest="edim", type=int, default=100, help='Set Embedding Dimension, default is 100')
parser.add_argument('-i', action="store",dest="idim", type=int, default=200, help='Set Bidirectional Internal Dimension, default is 200')
parser.add_argument('-d', action="store", dest="drop", type=restricted_float, default=0.1, help='Set Dropout in range 0-1, default is 0.1')
parser.add_argument('-stack', action="store_true", default=False, help='Use Stack CRF For Additional Bidirectional Layer')
parser.add_argument('-cnn', action="store_true", default=False, help='Use Image CNN Layer')
parser.add_argument('-v', action="store_true", default=False, help='Verbose mode during prediction')
parser.add_argument('-mask', action="store_true", default=False, help='Mask the padded chars during training')
parser.add_argument('-jiggle', action="store_true", default=False, help='Jiggle images during training')
parser.add_argument('-emnist', action="store_true", default=False, help='Use EMNIST vectors for training')
parser.add_argument('-emstack', action="store_true", default=False, help='Use EMNIST vectors stacked on embeddings for training')
parser.add_argument('-repeat', action="store", dest="repeat", type=int, default=0, help='# times to repeat training data with jiggled char images')
parser.add_argument('-epochs', action="store", dest="epochs", type=int, default=50, help='Number of epochs during training')
parser.add_argument('-batch', action="store", dest="batch", type=int, default=32, help='Batch size during training')
parser.add_argument('-gpu', action="store", dest="gpu", type=int, default=0, help='Set GPU id to run on, default is 0')
parser.add_argument('-models', action="store", dest="directory", default="models", help='directory to store the model files') 

# Subparsers for required test, train, or traintest
subparsers = parser.add_subparsers()
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

MAX_EPOCHS = args.epochs
MAX_LEN = args.maxlen
STACK = args.stack
EMBEDDING_DIM = args.edim #up by 50
INTERNAL_DIM = args.idim #Keep bigger
DROPOUT = args.drop #between 0-1
USE_VISUAL_CNN = args.cnn
JIGGLE_IMGS = args.jiggle
REPEAT_DATA = args.repeat
USE_MASK = args.mask
TEST_VERBOSE = args.v
BATCH_SIZE = args.batch   # 128 is too high, 64 is OK but maybe also too high, 32 good but slower
MODEL_DIRECTORY = args.directory
EMNIST = args.emnist
EM_STACK = args.emstack 
EM_SIZE = 62
EM_DICT = {}

# Check if model directory exists, make directory if not, also handled backslashes at end of specified directory
if MODEL_DIRECTORY[-1] != '/':
    MODEL_DIRECTORY = MODEL_DIRECTORY + '/'
if not os.path.isdir(MODEL_DIRECTORY):
    try:
        os.mkdir(MODEL_DIRECTORY)
    except OSError:
        print("Could not create specified directory")
    else:
        print("Created new model directory")

# Check if the unicode images directory exists.
if USE_VISUAL_CNN:
    if not os.path.isdir(images.LOAD_DIR):
        print("ERROR: path to the unicode image database doesn't exist: ", images.LOAD_DIR)
        sys.exit()

# If running in EMNIST mode -- didn't work as well as normal trained CNN
if EMNIST or EM_STACK:
    LOAD_DIR='/gpfs/scratch/nchamber/data/unicodeDB/emnist/emVectors.txt'
    if not os.path.isfile(LOAD_DIR):
        LOAD_DIR='../../../../corpora/unicodeDB/emnist/emVectors.txt'
    with open(LOAD_DIR,'r') as fn:
        for line in fn:
            k = line.strip().split(':')[0]
            v = np.array(line.strip().split(':')[1].split(',')).astype(np.float)
            EM_DICT.update({k:v})

GPU = args.gpu
print("GPU ID: ", GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)

# Print Parameters
print("EPOCHS        = " + str(MAX_EPOCHS))
print("BATCH N       = " + str(BATCH_SIZE))
print("EMBEDDING_DIM = " + str(EMBEDDING_DIM))
print("INTERNAL_DIM  = " + str(INTERNAL_DIM))
print("DROPOUT       = " + str(DROPOUT))
print("USE_CNN       = " + str(USE_VISUAL_CNN))
print("STACK         = " + str(STACK))
print("JIGGLE        = " + str(JIGGLE_IMGS))
print("REPEAT        = " + str(REPEAT_DATA))
print("USE_MASK      = " + str(USE_MASK))
print("EMNIST        = " + str(EMNIST))
print("EM_STACK      = " + str(EM_STACK))
print("*SUM* e%d-i%d-d%.2f-cnn%d-jig%d-stack%d-rep%d-mask%d" % 
      (EMBEDDING_DIM, INTERNAL_DIM, DROPOUT, int(USE_VISUAL_CNN), int(JIGGLE_IMGS), int(STACK), REPEAT_DATA, int(USE_MASK)))


class SentenceGetter(object):
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, t) for w, t in zip(s["Word"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None
            
def prettyPrintLabels(Y):
    for y in Y:
        print("[")
        for vals in y:
            print("  [ ", end='')
            for val in vals:
                print("%d" % (val), end=' ')
            print("]")
        print(" ]")

# Find the index at which the padding starts in an array
def findPadStart(myList, words, pad):
    i = len(myList) - 1
    while i >= 0 :
        if words[myList[i]-1] != pad:
            return i+1
        i -= 1
    return 0

# Translate an array of tags to a number
def tag2num(tags):
    number = ""
    for tag in tags:
        if "B" in tag:
            number += tag[1:]
    return number

# Compares two numbers and outputs the number of matching digits between them
def numCompare(num_1,num_2):
    correct = 0
    if len(num_1) >= len(num_2):
        for i in range(0,len(num_2)):
            if num_1[i] == num_2[i]:
                correct += 1
    else:
        for i in range(0,len(num_1)):
            if num_1[i] == num_2[i]:
                correct += 1
    return correct

# Compares two arrays of tags and output the number of matching tags
def tagCompare(tags_1,tags_2):
    tagCorrect = 0
    for i in range(0, len(tags_1)):
        if tags_1[i] == tags_2[i]:
            tagCorrect += 1
    return tagCorrect


def train(trainFile):
    data = pd.read_csv(trainFile, encoding="utf-8", delimiter="\t", quoting=csv.QUOTE_NONE)
    data = data.fillna(method="ffill")

    words = list(set(data["Word"].values))
    words.append("UNK")
    words.append("ENDPAD")
    n_words = len(words)

    tags = list(set(data["Tag"].values))
    tags.append('P') # the Padding tag at the end of strings
    n_tags = len(tags)

    print("TRAIN: n_tags=%d tags = %s" % (n_tags, tags))

    getter = SentenceGetter(data)
    sentences = getter.sentences
    print("Got %d sentences" % (len(sentences)))

    word2idx = {w: i + 1 for i, w in enumerate(words)}
    tag2idx = {t: i for i, t in enumerate(tags)}

    #print("test word2idx['o'] = " + str(word2idx['o']))
    #print("test word2idx['UNK'] = " + str(word2idx['UNK']))
    #print("test word2idx['ENDPAD'] = " + str(word2idx['ENDPAD']))

    X = [[word2idx[w[0]] for w in s] for s in sentences]
    X_tr = pad_sequences(maxlen=MAX_LEN, sequences=X, padding="post", value=word2idx['ENDPAD'])

    y = [[tag2idx[w[1]] for w in s] for s in sentences]
    y = pad_sequences(maxlen=MAX_LEN, sequences=y, padding="post", value=tag2idx["P"])

    #print("X_tr = " + str(X_tr))
    #print("y    = " + str(y))
    
    y_tr = [to_categorical(i, num_classes=n_tags) for i in y]

    #Save original input sequences
    if EM_STACK or EMNIST:
        OX_tr = X_tr
        emb_weights = np.random.randn(n_words+1, EM_SIZE)
        for word in words:
            try:
                key = str(hex(ord(word)).split('x')[1]).zfill(5)
            except:
                print(word, " could not be decoded to HEX")
                key = None

            # If we created the HEX and have this character in our lookup
            if key != None and key in EM_DICT:
                emb_weights[word2idx[word]] = EM_DICT.get(key)
            else:
                print(word," has no embedded weights.")
        
        em_input = Input(shape=(MAX_LEN,))    
        model = Embedding(input_dim=n_words+1, output_dim=EM_SIZE, weights=[emb_weights], input_length=MAX_LEN, mask_zero=True, trainable=False)(em_input)

        # Trainable dense layer on top of the untrainable EMNIST vectors.
        if EMNIST:
            model = TimeDistributed(Dense(EMBEDDING_DIM, activation="relu"))(model)
        else:
            em_stack = TimeDistributed(Dense(EMBEDDING_DIM, activation="relu"))(model)

         
        # Don't need padding, we'll set input to EM_SIZE and use a dense layer to expand to EMBEDDING_DIM

    # If using images for chars and the CNN is the first layer.
    if USE_VISUAL_CNN:
        stuff = []
        sent = ''
        for s in sentences:
            sent = ''
            for w in s:
                sent += w[0]
            stuff.append(sent)

        print("CNN getting %d texts" % (len(stuff)))

        (X_tr, maskvals) = images.charsToImages(stuff, MAX_LEN, jiggle=JIGGLE_IMGS, repeat=REPEAT_DATA)
        y_tr = images.repeatLabels(y_tr, REPEAT_DATA)
    
        # Chars as images
        (input, model) = imagesCNN.createCNN(MAX_LEN, 4, 8, 0, EMBEDDING_DIM)
        #(input, model) = imagesCNN.createCNN(MAX_LEN, 16, 16, 16, EMBEDDING_DIM)
        print(input.get_shape())

        # Masking
        if USE_MASK:
            input_mask = Input(shape=(MAX_LEN,1))
            model = multiply([model, input_mask])
            model = Masking(mask_value=0.0)(model)

        if EM_STACK:
            model = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=2))([model, em_stack]) 
            #dense layer here
    else:
        if not EMNIST:
            # Learned embeddings for the characters.
            input = Input(shape=(MAX_LEN,))
            model = Embedding(input_dim=n_words+1, output_dim=EMBEDDING_DIM, input_length=MAX_LEN, mask_zero=True)(input)

    # LSTM
    model = Bidirectional(LSTM(units=INTERNAL_DIM, return_sequences=True, recurrent_dropout=DROPOUT))(model)  # variational biLSTM

    if STACK:
        model = Bidirectional(LSTM(units=INTERNAL_DIM, return_sequences=True, recurrent_dropout=DROPOUT))(model)  # variational biLSTM
    model = TimeDistributed(Dense(50, activation="relu"))(model)  # a dense layer as suggested by neuralNer
    crf = CRF(n_tags)  # CRF layer
    out = crf(model)  # output

    if USE_VISUAL_CNN and USE_MASK:
        model = Model([input, input_mask], out)
        fitdata = [ X_tr, maskvals ]
    elif USE_VISUAL_CNN and EM_STACK:
        model = Model([input, em_input], out)
        fitdata = [ X_tr, OX_tr ]
    elif EMNIST:
        model = Model(em_input, out)
        fitdata = X_tr
    else:
        model = Model(input, out)
        fitdata = X_tr

    model.compile(optimizer="rmsprop", loss=crf_loss, metrics=[crf_viterbi_accuracy])
    print(model.summary())

    history = model.fit(fitdata, np.array(y_tr), batch_size=BATCH_SIZE, epochs=MAX_EPOCHS, validation_split=0.1,
                        callbacks=[EarlyStopping(monitor='val_loss', min_delta=.01, patience=5, restore_best_weights=True)],
                        verbose=1)

    print("Saving model...")
    sys.stdout.flush()
    modelPrefix = saveAll(model, [words,tags],MODEL_DIRECTORY)
    print("Model saved!",modelPrefix)

    # DEBUG LAYER OUTPUT
#    dalayer = model.get_layer('embedding_1').get_weights()
#    print('EMNIST Embedding weights:\n', dalayer)
#    dalayer = model.get_layer('time_distributed_1').get_weights()
#    print('EMNIST Dense weights:\n', dalayer)
#    dalayer = model.get_layer('bidirectional_1').get_weights()
#    print('LSTM weights:\n', dalayer)

    return modelPrefix


def test(modelPrefix, testFile):
    """
    Test a given model prefix on the given testFile path.
    The prefix is the prefix to files saved in the given prefix path.
    """
    if not MODEL_DIRECTORY == None:
        modelPrefix = MODEL_DIRECTORY + modelPrefix
    print("test with", modelPrefix)
    (tokenizer,model) = loadAllCRF(modelPrefix)
    #model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

    (texts,labels) = readUnicodeTSV(testFile)
    print("text length:", len(texts), "labels length:", len(labels))

    words = tokenizer[0]
    n_words = len(words)
    word2idx = {w: i + 1 for i, w in enumerate(words)}
    tags = tokenizer[1]

    # Save how long each obscured text is.
    lengths = list() 
    for t in texts:
        lengths.append(len(t))

    if USE_VISUAL_CNN:
        (X_te, maskvals) = images.charsToImages(texts, MAX_LEN)
    elif not USE_VISUAL_CNN or EM_STACK:
        X = []
        for s in texts:
            x = []
            for w in s:
                if w not in word2idx:
                    x.append(word2idx['UNK'])
                else: 
                    x.append(word2idx[w])
            X.append(x)

        if not USE_VISUAL_CNN:
            X_te = pad_sequences(maxlen=MAX_LEN, sequences=X, padding="post", value=word2idx['ENDPAD'])
        else:
            OX_te = pad_sequences(maxlen=MAX_LEN, sequences=X, padding="post", value=word2idx['ENDPAD'])

    overallLevCorrect = 0
    overallDigitCorrect = 0
    overallTotal = 0
    perfect = 0

    # Loop over phone numbers in the test set.
    for i in range(0,len(X_te)):
        prediction = list()
        obscure = list()

        if USE_VISUAL_CNN and USE_MASK:
            ps = model.predict( [ np.array([X_te[i]]), np.array([maskvals[i]]) ] )
        elif EM_STACK and USE_VISUAL_CNN:
            ps = model.predict( [ np.array([X_te[i]]), np.array([OX_te[i]]) ] )
        else:
            ps = model.predict(np.array([X_te[i]]))

        p = np.argmax(ps, axis=-1)
        true = labels
        padStart = lengths[i]  # Padding starts here.

        # Debug header
        if TEST_VERBOSE:
            print("{:15}||{}".format("Word", "Pred"))
            print(35 * "=")

        j = 0
        for pred in p[0]:
            prediction.append(tags[pred])
            obscure.append(texts[i][j])
            if TEST_VERBOSE:
                print(u'{:15}: {}'.format(texts[i][j], tags[pred]), end='')
                print("\t" + str(ps[0][j]))
                #if USE_VISUAL_CNN:
                #    images.prettyPrintCharImage(X_te[i][j])

            j += 1
            if j == padStart:
                break

        if TEST_VERBOSE:
            #print("X_te[i]:   " + str(X_te[i]))
            print("text:      ", end='')
            for ch in texts[i]:
                print(ch + '  ', end='')
            print("\npredicted: ", end='')
            for pred in p[0]:
                print(tags[pred] + ' ', end='')
                if len(tags[pred])==1:
                    print(' ', end='')
            print()

        # Digit Accuracy Calculation
        gold_num = str(true[i])
        pred_num = tag2num(prediction)

        if TEST_VERBOSE:
            print("gold: " + str(gold_num))
            print("pred: " + str(pred_num))

        digitCorrect = numCompare(gold_num,pred_num[0:10])
        if digitCorrect == 10:
            perfect += 1

        levWrong = levenshtein_distance(gold_num,pred_num)

        levCorrect = 10 - levWrong
        if levCorrect < 0:
            levCorrect = 0

        levAcc = float(levCorrect)/10
        digitAcc = float(digitCorrect)/10
        
        if TEST_VERBOSE:
            print("Lev Dist: {}".format(levWrong))
            print("Lev Accuracy: {:.2%}".format(levAcc))
            print("Digit Accuracy: {:.2%}".format(digitAcc))
            print
            print(35 * "-")
            print
        
        overallLevCorrect += levCorrect
        overallDigitCorrect += digitCorrect         
        overallTotal += 10

        if i % 1000 == 0:
            print(i, "...")

    overallDigitAcc = float(overallDigitCorrect)/overallTotal
    overallLevAcc = float(overallLevCorrect)/overallTotal
    overallPerfAcc = float(perfect)/len(X_te)

    print("Overall Lev Accuracy: {:.2%}".format(overallLevAcc))
    print("Overall Perfect Accuracy: {:.2%}".format(overallPerfAcc))
    print("Overall Digit Accuracy: {:.2%}".format(overallDigitAcc))



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
    modelPrefix = train(trainFile)

# TEST
if testFile:
    print("Model Prefix: ",modelPrefix)
    test(modelPrefix, testFile)






# ALTERNATE TESTING
def testOnTags(testFile):
    data = pd.read_csv(testFile, encoding="utf-8", delimiter="\t", quoting=csv.QUOTE_NONE)
    data = data.fillna(method="ffill")

    overallTagCorrect = 0
    overallTagTotal = 0
    overallDigitCorrect = 0
    overallDigitTotal = 0

    #for i in range(0,len(X_te)-1):
    for i in range(0,3):
        prediction = list()
        obscure = list()
        gold_std = list()
        p = model.predict(np.array([X_te[i]]))
        p = np.argmax(p, axis=-1)
        true = np.argmax(y_te[i], -1)
        print("{:15}||{:5}||{}".format("Word", "True", "Pred"))
        print(30 * "=")
    
        padStart = findPadStart(X_te[i], words[n_words-2])

        j = 0
        for w, t, pred in zip(X_te[i], true, p[0]):
            prediction.append(tags[pred])
            obscure.append(words[w-1])
            gold_std.append(tags[t])
            print(u'{:15}: {:5} {}'.format(words[w-1], tags[t], tags[pred]))
            j += 1
            if j == padStart:
                break

        # Tag Accuracy Calculation
        tagTotal = len(gold_std)
        tagCorrect = tagCompare(gold_std,prediction)

        print("Tag Total: {}", tagTotal)
        print("Tag Correct: {}", tagCorrect)

        tagAcc = float(tagCorrect)/tagTotal
        print("Tag Accuracy: {:.2%}".format(tagAcc))
    
        overallTagCorrect += tagCorrect
        overallTagTotal += tagTotal
    
        # Digit Accuracy Calculation
        gold_num = tag2num(gold_std)
        pred_num = tag2num(prediction)

        print(gold_num)
        print(pred_num)

        digitCorrect = numCompare(gold_num,pred_num)

        digitAcc = float(digitCorrect)/10
        print("Digit Accuracy: {:.2%}".format(digitAcc))
    
        overallDigitCorrect += digitCorrect
        overallDigitTotal += 10

        print("")
        print(30 * "-")
        print("")

        overallTagAcc = float(overallTagCorrect)/overallTagTotal
        overallDigitAcc = float(overallDigitCorrect)/overallDigitTotal

    print("Overall Tag Accuracy: {:.2%}".format(overallTagAcc))
    print("Overall Digit Accuracy: {:.2%}".format(overallDigitAcc))

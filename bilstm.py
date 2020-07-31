#
# This program is the main LSTM-based model.
# Do not call this directly, but instead use phoneTrainTest.py
# 
# This program originally adapted from a base bi-LSTM program here:
#      github.com/richliao/textClassifier/blob/master/textClassifierRNN.py
# Nice blog: richliao.github.io/supervised/classification/2016/12/26/textclassifier-RNN/
#
#
# Many additions to the original program above:
# (author - Nate Chambers)
# - TensorFlow backend works in full. Theano-specifics have been removed. 
# - Now works with Keras 2.
# - New attention mechanisms
# - Early stopping during training
# - Numerous functions for interacting with the model
# - Stacked LSTMs, conditioned prediction
#
#

import numpy as np
import pickle as pickle
from collections import defaultdict
import re
from helper import saveAll, numDigitMatch, readUnicodeFile, readUnicodeTSV, myevaluate, makeHeatmap
import imagesCNN
import images

from bs4 import BeautifulSoup

import sys
import os

#from theano import tensor as T, function, printing
import tensorflow as tf

#os.environ['KERAS_BACKEND']='theano'
os.environ['KERAS_BACKEND']='tensorflow'

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten, Concatenate, Lambda, ZeroPadding1D, Reshape, multiply, Multiply
# Nate removed 'Merge'
#from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, GRU, Bidirectional
from keras.models import Model
from keras.models import model_from_json, load_model

from keras import Sequential
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers

# New imports required
from keras.callbacks import EarlyStopping
import os.path

# Defaults
MAX_SEQUENCE_LENGTH = 70  # NATE: was 1000 for reviews
MAX_NB_WORDS = 2000
VALIDATION_SPLIT = 0.1
MAX_EPOCHS = 100
BATCH_SIZE = 32
# No need to ever change the assignments unless you change the dest option in the add_arguments function
EMBEDDING_DIM = 100
RNN_INTERNAL_DIM = 200
DROPOUT = .5
USE_ATTENTION = False
USE_SHARED_ATTENTION = False
CONDITIONED_OUTPUT = False
STACK_RNN = False
USE_PRETRAINING = False
USE_VISUAL_CNN = False
USE_CHAR_SIMILARITIES = False   # EXPERIMENTAL DON'T TURN TO TRUE
ONLY_CHAR_SIMILARITIES = False  # EXPERIMENTAL DON'T TURN TO TRUE
CHAR_SIM_LENGTH = 93  # 93 ASCII characters (threshParse.py)
JIGGLE_IMGS = False
REPEAT_DATA = 0
EXPAND_TOKENS = False
MODEL_DIRECTORY = None
HEATMAP = False


if __name__ == '__main__':
    print('Do not run this file directly.')

    
def printSettings():
    # Print settings
    print("--------------------------------")
    print("Max Sequence Length:     ", MAX_SEQUENCE_LENGTH)
    print("Max NB Word:             ", MAX_NB_WORDS)
    print("Validation Split:        ", VALIDATION_SPLIT)
    print("Max Training Epochs:     ", MAX_EPOCHS) 
    print("Batch Size:              ", BATCH_SIZE)
    print("Embedding Dimensions:    ", EMBEDDING_DIM)
    print("RNN Internal Dimensions: ", RNN_INTERNAL_DIM) 
    print("Dropout:                 ", DROPOUT)
    print("Use Attention:           ", USE_ATTENTION) 
    print("Use Shared Attention:    ", USE_SHARED_ATTENTION)
    print("Use Char Visual Sims:    ", USE_CHAR_SIMILARITIES)
    print("ONLY Char Visual Sims:   ", ONLY_CHAR_SIMILARITIES)
    print("Conditioned Output:      ", CONDITIONED_OUTPUT)
    print("Stack RNN:               ", STACK_RNN)
    print("Pre-Training:            ", USE_PRETRAINING)
    print("Images CNN Input:        ", USE_VISUAL_CNN)
    print("Images Jiggle chars:     ", JIGGLE_IMGS)
    print("Images Repeat Data:      ", REPEAT_DATA)
    print("Expanded Tokens:         ", EXPAND_TOKENS) 
    print("Model directory:         ", MODEL_DIRECTORY)
    print("*SUM* e%d-i%d-d%.2f-att%d-satt%d-cond%d-stack%d-cnn%d-jig%d-sims%d" % 
          (EMBEDDING_DIM, RNN_INTERNAL_DIM, DROPOUT, int(USE_ATTENTION), int(USE_SHARED_ATTENTION), int(CONDITIONED_OUTPUT), int(STACK_RNN), int(USE_VISUAL_CNN), int(JIGGLE_IMGS), int(USE_CHAR_SIMILARITIES)))
    print("--------------------------------")
    print()

    
# Attention GRU network		  
class AttLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get('normal')
        #self.input_spec = [InputSpec(ndim=3)]
        super(AttLayer, self).__init__(**kwargs)

    # NATE: Changed this 'build' function for Keras 2.
    #       Found the solution here: https://stackoverflow.com/questions/47554275/typeerror-not-a-keras-tensor-elemwiseadd-no-inplace-0

    # def build(self, input_shape):
    #     assert len(input_shape)==3
    #     self.W = self.init((input_shape[-1],))
    #     self.trainable_weights = [self.W]
    #     super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!

    def build(self, input_shape):
        assert len(input_shape)==3
        #print("INPUT_SHAPE", input_shape)
        #print("INPUT_SHAPE[-1]", input_shape[-1])
        self.W = self.add_weight(name='kernel', 
                                 shape=(input_shape[-1],),
                                 initializer='normal',
                                 trainable=True)
        #print "W", type(self.W), self.W, self.W.shape
        super(AttLayer, self).build(input_shape)
    
    def call(self, x, mask=None):
        #print("X SHAPE DIMENSIONS:", getShape(x))
        #print("W SHAPE DIMENSIONS:", getShape(self.W))
        
        # IMPORTANT: Keras dot() only works with Theano. TensorFlow's dot has
        #            different behavior, so we need to call TensorFlow directly with its
        #            tensordot() function that mirrors numpy's tensordot().
        # Nate: x is 3D batchx1000x200
        # Nate: W is 1D 200
        # Nate: eij is batchx1000
        if os.environ['KERAS_BACKEND'] == 'tensorflow':
            eij = K.tanh(tf.tensordot(x, self.W, axes=1)) # 50x1000x200 dot 200x1 ==> 50x1000
        else:
            eij = K.tanh(K.dot(x, self.W))   # 50x1000x200 dot 200x1 ==> 50x1000
            
        #print("eij SHAPE DIMENSIONS:", getShape(eij))

        # IMPORTANT: Original code used Keras dimshuffle() which TensorFlow doesn't have.
        #            But I realized this code's use of dimshuffle() wasn't shuffling
        #            anything and instead just adding a dimension. Keras' expand_dims()
        #            does that job and it works for both Theano/TF
        #
        # This computes the probabilities of the word weights for each review in the batch.
        # Nate: ai is batchx1000
        ai = K.exp(eij)
        # Nate: weights is batchx1000
        # Sums the columns for each row: (batchx1000) -> (batch)
        # Dimshuffle adds a fake x1 dimension so division works
        weights = ai/K.expand_dims(K.sum(ai, axis=1), axis=1)
            
        # Weights now has #rows which are the reviews in the batch.
        # Each row is 1000 long (#words in the sentence) with probabilities of word importance.
        #print("ai DIMENSIONS:", getShape(ai))
        #print("ai sum DIMENSIONS:", getShape(K.sum(ai, axis=1)))
        #print("weights DIMENSIONS", getShape(weights))
        
        # Nate: batchx1000x200 * batchx1000x1 ==> batchx1000x200
        weighted_input = x * K.expand_dims(weights, axis=-1)
        #print("weighted_input DIMENSIONS:", getShape(weighted_input))

        # Nate: batchx200 (sums out the middle dimension which was 1000)
        # Nate: I took out theano-specific line for Keras general code
        returnval = K.sum(weighted_input, axis=1)  
        #print("returnval DIMENSIONS", getShape(returnval))
        
        return returnval

    # NATE: Keras 2 requires "compute_output_shape" as the name.
    #    def get_output_shape_for(self, input_shape):
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


# Attention GRU network		  
class AttLayerNoSum(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get('normal')
        #self.input_spec = [InputSpec(ndim=3)]
        super(AttLayerNoSum, self).__init__(**kwargs)

    # NATE: Changed this 'build' function for Keras 2.
    #       Found the solution here: https://stackoverflow.com/questions/47554275/typeerror-not-a-keras-tensor-elemwiseadd-no-inplace-0

    # def build(self, input_shape):
    #     assert len(input_shape)==3
    #     self.W = self.init((input_shape[-1],))
    #     self.trainable_weights = [self.W]
    #     super(AttLayerNoSum, self).build(input_shape)  # be sure you call this somewhere!

    def build(self, input_shape):
        assert len(input_shape)==3
        #print("INPUT_SHAPE", input_shape)
        #print("INPUT_SHAPE[-1]", input_shape[-1])
        self.W = self.add_weight(name='kernel', 
                                 shape=(input_shape[-1],),
                                 initializer='normal',
                                 trainable=True)
        #print "W", type(self.W), self.W, self.W.shape
        super(AttLayerNoSum, self).build(input_shape)
    
    def call(self, x, mask=None):
        #print("X SHAPE DIMENSIONS:", getShape(x))
        #print("W SHAPE DIMENSIONS:", getShape(self.W))
        
        # IMPORTANT: Keras dot() only works with Theano. TensorFlow's dot has
        #            different behavior, so we need to call TensorFlow directly with its
        #            tensordot() function that mirrors numpy's tensordot().
        # Nate: x is 3D batchx1000x200
        # Nate: W is 1D 200
        # Nate: eij is batchx1000
        if os.environ['KERAS_BACKEND'] == 'tensorflow':
            eij = K.tanh(tf.tensordot(x, self.W, axes=1)) # 50x1000x200 dot 200x1 ==> 50x1000
        else:
            eij = K.tanh(K.dot(x, self.W))   # 50x1000x200 dot 200x1 ==> 50x1000
            
        #print("eij SHAPE DIMENSIONS:", getShape(eij))

        # IMPORTANT: Original code used Keras dimshuffle() which TensorFlow doesn't have.
        #            But I realized this code's use of dimshuffle() wasn't shuffling
        #            anything and instead just adding a dimension. Keras' expand_dims()
        #            does that job and it works for both Theano/TF
        #
        # This computes the probabilities of the word weights for each review in the batch.
        # Nate: ai is batchx1000
        ai = K.exp(eij)
        # Nate: weights is batchx1000
        # Sums the columns for each row: (batchx1000) -> (batch)
        # Dimshuffle adds a fake x1 dimension so division works
        weights = ai/K.expand_dims(K.sum(ai, axis=1), axis=1)
            
        # Weights now has #rows which are the reviews in the batch.
        # Each row is 1000 long (#words in the sentence) with probabilities of word importance.
        #print("ai DIMENSIONS:", getShape(ai))
        #print("ai sum DIMENSIONS:", getShape(K.sum(ai, axis=1)))
        #print("weights DIMENSIONS", getShape(weights))
        
        # Nate: batchx1000x200 * batchx1000x1 ==> batchx1000x200
        weighted_input = x * K.expand_dims(weights, axis=-1)
        #print("weighted_input DIMENSIONS:", getShape(weighted_input))

        # Nate: batchx200 (sums out the middle dimension which was 1000)
        # Nate: I took out theano-specific line for Keras general code
        #returnval = K.sum(weighted_input, axis=1)  
        #print "returnval DIMENSIONS", getShape(returnval)

        return weighted_input   # batchx1000x200

    # NATE: Keras 2 requires "compute_output_shape" as the name.
    #    def get_output_shape_for(self, input_shape):
    def compute_output_shape(self, input_shape):
        #return (input_shape[0], input_shape[-1])
        print("compute_output_shape!", input_shape, type(input_shape))
        return tuple(input_shape)


# Attention GRU network		  
class AttPositionOnlyLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get('normal')
        super(AttPositionOnlyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        #print("INPUT_SHAPE", input_shape)
        #print("INPUT_SHAPE[-2]", input_shape[-1])
        self.Wp = self.add_weight(name='kernelposition', 
                                  shape=(input_shape[-2],),
                                  initializer='normal',
                                  trainable=True)
        #print("Wp", type(self.Wp), self.Wp, self.Wp.shape)
        super(AttPositionOnlyLayer, self).build(input_shape)
    
    def call(self, x, mask=None):
        #print("types:", type(x), type(self.Wp))
        #print("X  SHAPE DIMENSIONS:", getShape(x))
        #print("Wp SHAPE DIMENSIONS:", getShape(self.Wp))
        #print("expanded DIMENSIONS: ", getShape(K.expand_dims(self.Wp, axis=-1)))
        #print("expanded DIMENSIONS 0: ", getShape(K.expand_dims(self.Wp, axis=0)))

        # Nate: batchx1000x200 * batchx1000x1 ==> batchx1000x200

        # THESE TWO ARE EQUIVALENT, RUN BY TF DIRECTLY (no keras)
        #weighted_input = tf.multiply(x,tf.expand_dims(self.Wp, axis=-1))
        weighted_input = x * tf.expand_dims(self.Wp, axis=-1)

        # Doesn't work, TF complains about dimensions.
        #weighted_input = Multiply(K.expand_dims(self.Wp, axis=-1))(x)
        print("weighted_input DIMENSIONS:", getShape(weighted_input), weighted_input.shape)

        # Nate: batchx200 (sums out the middle dimension which was 1000)
        # Nate: I took out theano-specific line for Keras general code
        returnval = K.sum(weighted_input, axis=1)  
        print("returnval DIMENSIONS", getShape(returnval))
        
        return returnval

    # NATE: Keras 2 requires "compute_output_shape" as the name.
    #    def get_output_shape_for(self, input_shape):
    def compute_output_shape(self, input_shape):
        print("AttPosOnly compute_output_shape", input_shape, type(input_shape))
        print("Returning:", (input_shape[0], input_shape[-1]))
        return (input_shape[0], input_shape[-1])
    

    
# Attention GRU network		  
class AttPositionLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get('normal')
        super(AttPositionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        print("INPUT_SHAPE", input_shape)
        print("INPUT_SHAPE[-2]", input_shape[-1])
        self.W = self.add_weight(name='kernel', 
                                 shape=(input_shape[-1],),
                                 initializer='normal',
                                 trainable=True)
        self.Wp = self.add_weight(name='kernelposition', 
                                  shape=(input_shape[-2],),
                                  initializer='normal',
                                  trainable=True)
        #print("W ", type(self.W), self.W, self.W.shape)
        #print("Wp", type(self.Wp), self.Wp, self.Wp.shape)
        super(AttPositionLayer, self).build(input_shape)
    
    def call(self, x, mask=None):
        #print("X  SHAPE DIMENSIONS:", getShape(x))
        #print("W  SHAPE DIMENSIONS:", getShape(self.W))
        #print("Wp SHAPE DIMENSIONS:", getShape(self.Wp))
        
        # IMPORTANT: Keras dot() only works with Theano. TensorFlow's dot has
        #            different behavior, so we need to call TensorFlow directly with its
        #            tensordot() function that mirrors numpy's tensordot().
        # Nate: x is 3D batchx1000x200
        # Nate: W is 1D 200
        # Nate: eij is batchx1000
        if os.environ['KERAS_BACKEND'] == 'tensorflow':
            eij = K.tanh(tf.tensordot(x, self.W, axes=1)) # 50x1000x200 dot 200x1 ==> 50x1000
        else:
            eij = K.tanh(K.dot(x, self.W))   # 50x1000x200 dot 200x1 ==> 50x1000
            
        #print("eij SHAPE DIMENSIONS:", getShape(eij))

        # IMPORTANT: Original code used Keras dimshuffle() which TensorFlow doesn't have.
        #            But I realized this code's use of dimshuffle() wasn't shuffling
        #            anything and instead just adding a dimension. Keras' expand_dims()
        #            does that job and it works for both Theano/TF
        #
        # This computes the probabilities of the word weights for each review in the batch.
        # Nate: ai is batchx1000
        ai = K.exp(eij)
        # Nate: weights is batchx1000
        # Sums the columns for each row: (batchx1000) -> (batch)
        # Dimshuffle adds a fake x1 dimension so division works
        weights = ai/K.expand_dims(K.sum(ai, axis=1), axis=1)

        # Position weights
        #print("weights DIMENSIONS:", getShape(weights))
        weights = self.Wp * weights
        #print("weights DIMENSIONS:", getShape(weights))
        
        # Weights now has #rows which are the reviews in the batch.
        # Each row is 1000 long (#words in the sentence) with probabilities of word importance.
        #print("ai DIMENSIONS:", getShape(ai))
        #print("ai sum DIMENSIONS:", getShape(K.sum(ai, axis=1)))
        #print("weights DIMENSIONS", getShape(weights))
        
        # Nate: batchx1000x200 * batchx1000x1 ==> batchx1000x200
        weighted_input = x * K.expand_dims(weights, axis=-1)
        #print("weighted_input DIMENSIONS:", getShape(weighted_input))

        # Nate: batchx200 (sums out the middle dimension which was 1000)
        # Nate: I took out theano-specific line for Keras general code
        returnval = K.sum(weighted_input, axis=1)  
        #print("returnval DIMENSIONS", getShape(returnval))
        
        return returnval

    # NATE: Keras 2 requires "compute_output_shape" as the name.
    #    def get_output_shape_for(self, input_shape):
    def compute_output_shape(self, input_shape):
        print("AttPos compute_output_shape", input_shape, type(input_shape))
        print("Returning:", (input_shape[0], input_shape[-1]))
        return (input_shape[0], input_shape[-1])

    
########################################
# LOAD THE MODEL
def loadAll(modelname):
    
    # load json and load the model architecture
   # if MODEL_DIRECTORY != None:
    #    modelname = MODEL_DIRECTORY + modelname
    json_file = open(modelname+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json, custom_objects={'AttLayer': AttLayer, 'AttLayerNoSum': AttLayerNoSum, 'AttPositionLayer': AttPositionLayer, 'AttPositionOnlyLayer': AttPositionOnlyLayer, 'ZeroPadding1D': ZeroPadding1D})
    
    # load the learned weights into this model
    model.load_weights(modelname+'.h5')
    print("Loaded model from disk")
    
    # Load the tokenizer word->ID
    if os.path.isfile(modelname+'.tokenizer.pickle'):
        with open(modelname+'.tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
    else:
        print("WARNING: tokenizer not found in loadAll("+modelname+")")
        tokenizer = None

    return (tokenizer,model)
        
# FOR DEBUGGING
def printWordIndex(index):
    count = 0
    max = 0
    newdict = dict()
    for word, jj in list(index.items()):
        newdict[jj] = word
        count = count + 1
        if jj > max:
            max = jj
    print("Counted", count, "in word index!")
    print("Max seen was", max)

    for jj in range(1,len(list(index.items()))):
        print(jj, newdict[jj])

# FOR DEBUGGING CHAR EMBEDDINGS
# Expects a double array, each row is a char's embedding
def compareEmbeddings(embeds, index):
    revdict = dict()
    for word, jj in list(index.items()):
        revdict[jj] = word

    for ii in range(1,len(embeds)):
        best = -1.0
        x = embeds[ii]
        # Find most similar
        for jj in range(1,len(embeds)):
            if ii != jj:
                y = embeds[jj]
                dot = np.dot(x,y)
                x_modulus = np.sqrt((x*x).sum())
                y_modulus = np.sqrt((y*y).sum())
                cos = dot / x_modulus / y_modulus
                if cos > best:
                    besti = jj
                    best = cos

        print("Most similar to", ii, revdict[ii], 'is', revdict[besti], 'at', best)

def getShape(x):
    if os.environ['KERAS_BACKEND'] == 'tensorflow':
        return K.int_shape(x)
    else:
        # Theano doesn't have shape information, only how many dimensions.
        return x.type.ndim

# Predict a phone from a single obscured string.
def predictPhone(tokenizer, model, obscured):
    # Prepare the input.
    obscured = obscured.split()  # "s e v e n"
    arr = list()
    arr.append(obscured)
    sequences = tokenizer.texts_to_sequences(arr)
    sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
        
    # Predict!
    yprobs = model.predict(sequences)
    
    # Build the phone number from the digit guesses
    phone = 0
    index = 0
    theprobs = list()
    for daprobs in yprobs:
        if index > 0:
            phone = phone * 10
        index += 1
        phone = phone + daprobs[0].argmax(axis=-1)
        theprobs.append(daprobs[0].max(axis=-1))

    #print "predictPhone:", phone, "\t", theprobs
    return (phone, theprobs)


# Interactive user testing!
def interactiveUser(tokenizer, model):
    while True:
        sys.stdout.flush()
        obscured = input("Input: ")     # "seven"
        obscured = obscured.split()  # "s e v e n"
        arr = list()
        arr.append(obscured)
        if USE_VISUAL_CNN:
            (sequences, maskvals) = images.charsToImages(arr, MAX_SEQUENCE_LENGTH)
        else:
            sequences = tokenizer.texts_to_sequences(arr)
            sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
        
        # Predict!
        yprobs = model.predict(sequences)

        print(obscured, '-->', end=' ')
        # Print guesses
        for daprobs in yprobs:
            print(daprobs[0].argmax(axis=-1), end=' ')
        print('\t')
        # Print probs
        for probarr in yprobs:
            for prob in probarr[0]:
                print('%.3f' % prob, end=' ')
            print('||', end=' ')
        print()

def phonesToCategoricalLabelVecs(phones):
    '''
    phones: list of int phone numbers (e.g., 4109342302)
    returns: Each phone is split into digits, and each digit is a one-hot vector of 10.
    '''
    # 10 label lists.
    allLabels = list()
    for i in range(0,10):
        allLabels.append(list())
    for phone in phones:
        for i in range(9,-1,-1):
            allLabels[i].append(phone%10)
            phone = int(phone / 10)

    # Turn digits into 10-binary vectors
    allCatLabels = list()
    for i in range(0,10):
        allCatLabels.append(to_categorical(np.asarray(allLabels[i])))
    return allCatLabels

def test(modelPrefix, testFile, interactive=False):
    """
    Load a model from file, and compute accuracy on the given data file.
    """

    if not MODEL_DIRECTORY == None:
        modelPrefix = MODEL_DIRECTORY + modelPrefix
    print("test with", modelPrefix)
    (tokenizer,model) = loadAll(modelPrefix)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

    # DEBUG: Print weights
    if USE_ATTENTION or USE_SHARED_ATTENTION:
        data = []  #2D array for the heatmap
        for i in range(0,10):
            # Print the position weights!
            #print("att"+str(i)+" layer position weights:", end=' ')
            
            dalayer = model.get_layer("att"+str(i)).get_weights()
            if USE_SHARED_ATTENTION:
                print(dalayer[0])
            elif USE_ATTENTION:
                print(dalayer[1])
                s = "" 
                layerWeights = []
                #if i != 9 : #which weights make heatmap with 
                #    continue
                for w in dalayer[1]:
                    s += str(w) + ','
                    #if i == 3 or i == 6: #invert these rows
                    #    w = w * -1 
                    layerWeights.append(w)
                data.append(layerWeights) 
                #print(s) #weights in csv form
        if HEATMAP:
            makeHeatmap(data,height=5,width=15,filename="heatmap.png")

    # Load the test file (CNN images or char2vec)
    if USE_VISUAL_CNN:
        (texts,labels) = readUnicodeTSV(testFile)
        (data, maskvals) = images.charsToImages(texts, MAX_SEQUENCE_LENGTH)
        print("Loaded all visual chars for phones.")
    elif ONLY_CHAR_SIMILARITIES:
        (texts,labels) = readUnicodeTSV(testFile)
        data = images.charsToCharSims(texts, MAX_SEQUENCE_LENGTH)
    else:
        (texts,labels) = readUnicodeTSV(testFile)
        sequences = tokenizer.texts_to_sequences(texts)
        data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    if USE_CHAR_SIMILARITIES:
        sims = images.charsToCharSims(texts, MAX_SEQUENCE_LENGTH)
        data = [ data, sims ]

    print("text length:", len(texts), "labels length:", len(labels))
    #print "label ex:", labels[1]
    myevaluate(model, data, labels, debug=True)
        
    # The columns are now the phone numbers (labels dim: 10 x numdatums)
    #labels = phonesToCategoricalLabelVecs(labels)
    #(loss,acc) = model.evaluate(data, labels)
    #sall = model.evaluate(data, labels)
    #print "Accuracy:",acc
    #print "All:",sall

    if interactive:
        interactiveUser(tokenizer, model)

def zeropad(x):
    '''
    Add zeros to the front of every row in the given 2d tensor.
    '''
    #print("x shape", x.shape)
    x = K.expand_dims(x,axis=-1)  # adds a 1x dimension
    #print("x shape", x.shape)
    z = ZeroPadding1D(padding=(10,0))(x)  # requires 3d input tensor: add 10 zeros to front, 0 to end
    #print("z shape", z.shape)
    z = K.squeeze(z, -1)  # remove that 1x dimension
    #print("z shape", z.shape)
    return z

def zeropad_output_shape(input_shape):
    shape = list(input_shape)
    shape[1] = shape[1]+10
    return tuple(shape)

def layerconcat(x):
    con = K.concatenate([x[0], x[1]], axis=1)
    return con

def layerconcat2(x):
    con = K.concatenate([x[0], x[1]], axis=2)
    return con

def setWeights(word_index):
    '''
    This uses pretrained embeddings from file to initialize the char 
    embeddings.
    '''

    embedding_dir = '/gpfs/home/griswold/navynlp/scripts/phone/'

    embeddings_index = {}
    f = open(os.path.join(embedding_dir, 'embedding_'+ str(EMBEDDING_DIM) +'.txt'))
    for line in f:
        values = line.split(", ")
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print(('Found %s word vectors.' % len(embeddings_index)))

    #embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))

    for word, i in list(word_index.items()):
        embedding_vector = embeddings_index.get("u"+repr(word))

        if embedding_vector is None:
            embedding_vector = embeddings_index.get(repr(word))
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        else:
            embedding_matrix[i] = embedding_vector
    print("Embedding matrix ...")
    print(embedding_matrix)
    
    return embedding_matrix


def train(trainFile):
    """
    Given a path to a tab-separated data file, this trains the RNN with attention.
    """
    # TRAINING CODE FOLLOWS

    # Characters represented by 34x34 images.
    if USE_VISUAL_CNN:
        (texts,labels) = readUnicodeTSV(trainFile)
        (data, maskvals) = images.charsToImages(texts, MAX_SEQUENCE_LENGTH, jiggle=JIGGLE_IMGS, repeat=REPEAT_DATA)
        print("Loaded all visual chars for phones.")

    # Characters represnted by unique ID. (normal word2vec style)
    else:
        (texts,labels) = readUnicodeTSV(trainFile)
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='\n', lower=False, char_level=True, oov_token='UNK')

        if EXPAND_TOKENS:
            text = readUnicodeFile("/gpfs/scratch/nchamber/data/processed/backpage-no-oneliners.txt")
            tokenizer.fit_on_texts(text)
        else:
            tokenizer.fit_on_texts(texts)

        sequences = tokenizer.texts_to_sequences(texts)
        data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

        printWordIndex(tokenizer.word_index)
        word_index = tokenizer.word_index  # starts at 1 (not 0)
        print(('Found %s unique tokens.' % len(word_index)))
        #print(data[0])
        #print(data[1])

#    print("orig labels: " + str(len(labels)))
#    print("orig labels[0] = " + str(labels[0]))

    # 10 label lists.
    # Convert gold unicode phones to gold integer phones
    for i in range(0,len(labels)): # might not need this cast at all
        labels[i] = int(labels[i])
    labels = images.repeatGoldDigits(labels, REPEAT_DATA)
    allCatLabels = phonesToCategoricalLabelVecs(labels)
    numLabels = allCatLabels[0].shape[1]
#    print(('Shape of data tensor:', data.shape))
#    print(('Shape of label tensor:', allCatLabels[0].shape))

    # Shuffle the input data.
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    for i in range(0,10):
        allCatLabels[i] = allCatLabels[i][indices]
    # Shuffle the texts list to match. Convert to Numpy array for ease.
    texts = np.array(texts)
    texts = texts[indices]

    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
    x_train = data[:-nb_validation_samples]
    x_train_texts = texts[:-nb_validation_samples]
    x_val = data[-nb_validation_samples:]
    x_val_texts = texts[-nb_validation_samples:]
    y_train = list()
    y_val = list()
    for i in range(0,10):
        y_train.append(allCatLabels[i][:-nb_validation_samples])
        y_val.append(allCatLabels[i][-nb_validation_samples:])
        #y_train = [ labels1[:-nb_validation_samples], labels2[:-nb_validation_samples], labels3[:-nb_validation_samples] ]
        #y_val = [ labels1[-nb_validation_samples:], labels2[-nb_validation_samples:], labels3[-nb_validation_samples:] ]

#    print('Training and validation set number of labels')  # 4th digit
#    print(y_train[4].sum(axis=0))
#    print(y_val[4].sum(axis=0))

    # ----------------------------------------------------------------
    # Create the model architecture.        

    # Chars as images
    if USE_VISUAL_CNN:
        (sequence_input, embedded_sequences) = imagesCNN.createCNN(MAX_SEQUENCE_LENGTH, 4, 8, 0, EMBEDDING_DIM)

    # Chars as similarity vecs to the main 93 ASCII characters
    elif ONLY_CHAR_SIMILARITIES:
        sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH, CHAR_SIM_LENGTH))
        embedded_sequences = sequence_input  # no further action needed, we input directly the values we need

        x_train = images.charsToCharSims(x_train_texts, MAX_SEQUENCE_LENGTH)
        x_val = images.charsToCharSims(x_val_texts, MAX_SEQUENCE_LENGTH)

    # Chars as embeddings
    else:
        sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        if USE_PRETRAINING:
            embedding_matrix = setWeights(word_index)
            embedding_layer = Embedding(len(word_index) + 1,
                                        EMBEDDING_DIM,
                                        weights=[embedding_matrix],
                                        input_length=MAX_SEQUENCE_LENGTH,
                                        trainable=True)
        else:
            embedding_layer = Embedding(len(word_index) + 1,
                                        EMBEDDING_DIM,
                                        input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=True)
        embedded_sequences = embedding_layer(sequence_input)

    # sequence_input = Tensor("input_1:0", shape=(?, 70), dtype=int32)
    # embedded_sequences = Tensor("embedding_1/embedding_lookup/Identity:0", shape=(?, 70, 100), dtype=float32)
    print("sequence_input =", sequence_input)
    print("embedded_sequences =", embedded_sequences)

    # If concatenating a vector of char similarities to each char's embedding.
    if USE_CHAR_SIMILARITIES:
        # Model params.
        charsim_input = Input(shape=(MAX_SEQUENCE_LENGTH,CHAR_SIM_LENGTH))
        embedded_sequences = Lambda(layerconcat2)([embedded_sequences,charsim_input])
        # Actual data input... x_train, y_train, validation_data=(x_val, y_val)
#        print("len(x_train): ", len(x_train), " len(x_train[0]): ", len(x_train[0]))
#        print("x_train[0]: ", x_train[0])

        sims = images.charsToCharSims(x_train_texts, MAX_SEQUENCE_LENGTH)
        sims_val = images.charsToCharSims(x_val_texts, MAX_SEQUENCE_LENGTH)
#        print("texts[0]: ", x_train_texts[0])
#        print("sims[0]:  ", sims[0])
#        print("sims[0]:  ", end='')
#        k = 0
#        for sim in sims[0]:
#            print("  " + str(k) + "=" + str(sim))
#            k += 1

    # Two layers of GRUs
    if STACK_RNN:
        embedded_sequences = Bidirectional(GRU(RNN_INTERNAL_DIM, return_sequences=True, dropout=DROPOUT))(embedded_sequences)

    # Prediction uses the previous predicted digit.
    if CONDITIONED_OUTPUT:
        l_atts  = list()
        preds = []

        # One shared RNN Attention, but each digit has its own Position Attention
        # DOES NOT WORK, TF COMPLAINS!
        if USE_SHARED_ATTENTION:
            l_gru = Bidirectional(GRU(RNN_INTERNAL_DIM, return_sequences=True, dropout=DROPOUT))(embedded_sequences)      
            l_att = AttLayerNoSum(name="att")(l_gru)
            for i in range(0,10):
                atpp = AttPositionOnlyLayer(name="att"+str(i))(l_att)
                l_atts.append(attp)
                
        # DOES NOT WORK, TF COMPLAINS!
        elif USE_ATTENTION:
            l_gru = Bidirectional(GRU(RNN_INTERNAL_DIM, return_sequences=True, dropout=DROPOUT))(embedded_sequences)
            for i in range(0,10):
                l_atts.append(AttPositionLayer(name="att"+str(i))(l_gru))

        # No attention, just the GRU's final state.
        else:
            l_gru = Bidirectional(GRU(RNN_INTERNAL_DIM, dropout=DROPOUT))(embedded_sequences)
            for i in range(0,10):
                l_atts.append(l_gru)

        # Append the previous digit prediction to the current digit classifier.
        for i in range(0,10):
            if i == 0:
                con = Lambda(zeropad, output_shape=zeropad_output_shape)(l_atts[i])
            else: # can't just K.concatenate() here, seems to require a Lambda layer to do it
                con = Lambda(layerconcat)([pred,l_atts[i]])

            pred = Dense(numLabels, activation='softmax')(con)
            preds.append(pred)
    
    # Use attention over each RNN state?
    elif USE_ATTENTION or USE_SHARED_ATTENTION:
        l_gru = Bidirectional(GRU(RNN_INTERNAL_DIM, return_sequences=True, dropout=DROPOUT))(embedded_sequences)
        l_atts  = list()
        preds = []
        
        # One shared RNN Attention, but each digit has its own Position Attention
        if USE_SHARED_ATTENTION:
            l_att = AttLayerNoSum(name="att")(l_gru)
            for i in range(0,10):
                l_atts.append(AttPositionOnlyLayer(name="att"+str(i))(l_att))
                preds.append(Dense(numLabels, activation='softmax')(l_atts[i]))

        # Each digit has its own GRU Attention and Position Attention
        else:
            for i in range(0,10):
                l_atts.append(AttPositionLayer(name="att"+str(i))(l_gru))
                preds.append(Dense(numLabels, activation='softmax')(l_atts[i]))

    # elif CONDITIONED_OUTPUT:
    #     l_gru = Bidirectional(GRU(RNN_INTERNAL_DIM, dropout=DROPOUT))(embedded_sequences)
    #     preds = []
    #     for i in range(0,10):
    #         if i == 0:
    #             con = Lambda(zeropad, output_shape=zeropad_output_shape)(l_gru)
    #         else: # can't just K.concatenate() here, seems to require a Lambda layer to do it
    #             con = Lambda(layerconcat)([pred,l_gru])

    #         pred = Dense(numLabels, activation='softmax')(con)
    #         preds.append(pred)
            
    # No attention. Just use the final RNN state.
    else:
        l_lstm = Bidirectional(LSTM(RNN_INTERNAL_DIM, dropout=DROPOUT))(embedded_sequences)
        preds = []
        for i in range(0,10):
            preds.append(Dense(numLabels, activation='softmax')(l_lstm))
    
    #print("input =",sequence_input)
    #print("preds =",preds)

    if USE_CHAR_SIMILARITIES:
        model = Model([sequence_input, charsim_input], preds)
        fitdata = [ x_train, sims ]
        fitdata_val = [ x_val, sims_val ]
    else:
        model = Model(sequence_input, preds)
        fitdata = x_train
        fitdata_val = x_val

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
    model.summary()

    # Model training.
    model.fit(fitdata, y_train, validation_data=(fitdata_val, y_val),
              epochs=MAX_EPOCHS, 
              batch_size=BATCH_SIZE,
              callbacks=[EarlyStopping(monitor='val_loss', min_delta=.001, patience=6)])
    print("Model finished fitting!")
    sys.stdout.flush()

    # DEBUGGING
    if not USE_VISUAL_CNN and not ONLY_CHAR_SIMILARITIES:
        print(tokenizer.word_index)
        compareEmbeddings(embedding_layer.get_weights()[0], tokenizer.word_index)
        sys.stdout.flush()
    else:
        tokenizer = None

    # Save the model and tokenizer to disk.
    print("Saving model...")
    sys.stdout.flush()
    modelPrefix = saveAll(model, tokenizer,MODEL_DIRECTORY)
    print("Model saved!",modelPrefix)

    # DEBUGGING ONLY: Print weights 
    if False:
        if USE_ATTENTION or USE_SHARED_ATTENTION:
            for i in range(0,10):
                # Print the position weights!
                print("att"+str(i)+" layer position weights:", end=' ')
                dalayer = model.get_layer("att"+str(i)).get_weights()
                if USE_SHARED_ATTENTION:   print(dalayer[0])
                elif USE_ATTENTION:        print(dalayer[1])

    # Interactive user testing!
    # interactiveUser(tokenizer, model)
    return modelPrefix

#
# Helper functions.
#
# @author Nate Chambers
#
# ----------------------------------------
# NOTES ON UNICODE
#
# Python 2 has two types: string and unicode
# Python 3 handles unicode in the single string type.
#
# Pandas will read files as UTF-8 if you set encoding='utf-8'
# - This returns variables with unicode type (not string)
#
# Unicode types have unicode functions that give unicode types back:
#   - strip()
#   - [start:end]
#
# Printing unicode types to a terminal will print as UTF-8 just fine.
#
# Piping python output to a file will NOT print as UTF-8. It will
# default to ASCII and crash. This means you can run your code without
# error, but then it will crash when piping to a file. Fix this by
# setting the environment variable:
#   PYTHONIOENCODING=UTF-8
# ----------------------------------------
# 

import pandas as pd
import codecs
import os.path
#import cPickle as pickle   #Python2
import pickle
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.models import model_from_json
from keras_contrib.layers import CRF


def readUnicodeFile(path):
    text = []
    f = codecs.open(path, encoding='utf-8')
    for line in f:
        text.append(line)
    return text

def readUnicodeTSV(path):
    texts = []
    labels = []
    ii = 0
    f = codecs.open(path, encoding='utf-8')
    for line in f:
        if ii > 0:
            line = line.strip()
            split = line.find('\t')
            label = line[0:split]
            text = line[split+1:]
            labels.append(label)
            texts.append(text)
        ii += 1
    return (texts,labels)

## WARNING: doesn't allow 0's in the guess at the start of guess
##          since these are ints
## MAYBE GET RID OF THIS IN ALL CODE
##
def numDigitMatchInts(gold, guess):
    '''
    Returns the number of digits in the two integers that match.
    Expects two integers of the same 'size'.
    '''
    gold = int(gold)
    guess = int(guess)
    correct = 0
    for i in range(0,10):
        if gold % 10 == guess % 10:
            correct += 1
        gold = gold // 10
        guess = guess // 10
    return correct;

def numDigitMatch(gold, guess):
    '''
    Returns the number of digits in the two strings that match.
    If the guess is less than 10, it treats the missing as incorrect.
    '''
    gold = str(gold)
    guess = str(guess)
    guessN = len(guess)
    end = min(guessN,10)

    correct = 0
    for i in range(0,end):
        if gold[i] == guess[i]:
            correct += 1
    return correct;

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
            phone = phone // 10
            
    # Turn digits into 10-binary vectors
    allCatLabels = list()
    for i in range(0,10):
        allCatLabels.append(to_categorical(np.asarray(allLabels[i]), num_classes=10))

    return allCatLabels



#############################################################
# Credit for function:
# Project: chalktalk_docs   Author: loremIpsum1771   File: versioning.py
# Found at: https://www.programcreek.com/python/example/94974/Levenshtein.distance
def levenshtein_distance(a, b):
    """Return the Levenshtein edit distance between two strings *a* and *b*."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    if not a:
        return len(b)
    previous_row = range(len(b) + 1)
    for i, column1 in enumerate(a):
        current_row = [i + 1]
        for j, column2 in enumerate(b):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (column1 != column2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def myevaluate(model, data, labels, debug=False):
    '''
    Evaluates the given phone number model by calling predict on the given data.
    Compares predictions to the labels which should be gold phone numbers.
    '''
    predictions = model.predict(data) # 10xnumDatums
    print("PREDICTIONS: ", type(predictions), len(predictions), len(predictions[0]))

    # Make the list of guesses for all datums.
    guesses = list()
    for d in range(0,len(labels)):
        gold = labels[d]
        guess = ""
        for i in range(0,10):
            digit = np.argmax(predictions[i][d])
            guess = guess + str(digit)
        guesses.append(guess)
  
    # Grab digit guesses from the probabilities and compute accuracy metrics.
    perfect = 0
    totalCorrect = 0
    levTotalCorrect = 0
    for d in range(0,len(labels)):
        gold = labels[d]
        guess = guesses[d]
        numcorrect = numDigitMatch(gold, guess)
        totalCorrect += numcorrect

        digitWrong = levenshtein_distance(str(gold), str(guess))
        digitCorrect = 10 - digitWrong
        if digitCorrect < 0:
            digitCorrect = 0

        levTotalCorrect += digitCorrect

        if debug:
            print("**", gold, "-->", guess, numcorrect)
        if numcorrect == 10:
            perfect += 1

    #displays number of digits guesses right and wrong
    if False:
        D = [[0 for i in range(10)] for i in range(10)]
        for i in range(len(labels)):
            gold = labels[i]
            guess = guesses[i]
            if len(str(guess)) != 10:
                continue    
            #print(gold)
            for d in str(gold):
                D[int(d)][int(str(guess)[str(gold).index(str(d))])] += 1        
        googleDoc = ""
        for i in range(10):
            print('--- ' + str(i) + ' ---')
            total = 0
            incorrect = 0
            s = "" 
            for j in range(10):
                #print(str(j) + ': ' + 'x'*D[i][j])
                print(str(j) + ': ' + str(D[i][j]))
                s += str(D[i][j]) + ','
                total += D[i][j] 
                if i != j:
                    incorrect += D[i][j]
            percent_incorrect = str(round(float(incorrect)/total*100,2))
            s += percent_incorrect
            googleDoc += s + '\n'
            print("% incorrect: " + percent_incorrect)
        print('\n' + "copy paste for google sheets:")
        print(googleDoc)

    print("Overall Lev Accuracy: %.2f" % (100*levTotalCorrect/(10.0*len(labels))) + "%")
    print("Overall Perfect Accuracy: %.2f" % (100*float(perfect)/len(labels)) + "%")
    print("Overall Digit Accuracy: %.2f" % (100*totalCorrect/(10.0*len(labels))) + "%")



########################################
# LOAD THE MODEL
def loadModel(modelname):
    # Load json and load the model architecture
    json_file = open(modelname+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    
    # Load the learned weights into this model
    model.load_weights(modelname+'.h5')
    print("Loaded model from disk")

    return model

########################################
# SAVE THE MODEL
# https://machinelearningmastery.com/save-load-keras-deep-learning-models/
def saveModel(model):
    # Decide on the model name.
    count = 1
    outputName = "model"
    while os.path.isfile(outputName+'.h5'):
        outputName = 'model' + str(count)
        count+=1
        
    # serialize model to JSON
    model_json = model.to_json()
    with open(outputName + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF
    model.save_weights(outputName + ".h5")
    print(("Saved model to disk as: " + outputName))

    # Save the model as one file? (redundant with above)
    model.save(outputName + ".full")

    return outputName


########################################
# LOAD THE MODEL
def loadAllCRF(modelname):

    # load json and load the model architecture
    json_file = open(modelname+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json, custom_objects={'CRF': CRF})

    # load the learned weights into this model
    model.load_weights(modelname+'.h5')
    print("Loaded model from disk")

    # Load the tokenizer word->ID
    with open(modelname+'.tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    return (tokenizer,model)

########################################
# SAVE THE MODEL
# https://machinelearningmastery.com/save-load-keras-deep-learning-models/
def saveAll(model, tokenizer, directory):
    # Decide on the model name.
    count = 1
    outputName = "model"
    outputNameWithDir = outputName
    outputNameWithDir = directory + "model"

    while os.path.isfile(outputNameWithDir+'.h5'): #outPutNameWithDir will have the directory appended infront if the tag is used, otherwise its jsut the model name
        outputName = 'model' + str(count)
        outputNameWithDir = directory + 'model' + str(count)
        count+=1
    modelName = outputName # modelName = model file name without directory
    outputName = directory + outputName

    # serialize model to JSON
    model_json = model.to_json()
    with open(outputName + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF
    model.save_weights(outputName + ".h5")
    print("Saved model to disk as: " + outputName)

    # Save the tokenizer word index.
    if not tokenizer == None:
        with open(outputName + '.tokenizer.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Saved tokenizer to disk")

    # Save the model as one file? (redundant with above)
    model.save(outputName + ".full")

    return modelName

def makeHeatmap(data,height,width,filename):
    '''
    Makes a heatmap with the data provided
    Input:
        data: 2D array containing the values, list , ex. 10 rows & 70 cols is a list containing 10 lists of 70 values
        height,width: size of heatmap in inches, int
        filename: name of file to save heatmap, string
    '''
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(width,height), gridspec_kw=dict(top=1-.04, bottom=.04))
    ax = sns.heatmap(data, ax=ax, cmap="seismic_r", center = 0, cbar = False)
    plt.axis('off')
    plt.savefig(filename)


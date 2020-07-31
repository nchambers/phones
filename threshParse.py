#
# EXPERIMENTAL: DON'T USE
#
# Old code used on the RNN experiments to add some char similarity
# scores to the learner.
# 

import os
import io
from collections import defaultdict

fname="/gpfs/scratch/nchamber/data/char-similarities.txt"
simvalues=defaultdict(list)

def valueDict(method='real', threshold=.70):
    global simvalues

    N = 0
    with open(fname, 'r') as f:
        for line in f.readlines():
            key = line[29:34] #Unicode character from library
            char = line[19:24] #Ascii character for comaprison
            val = line[36:40] #Similarity as decimal percentage

            #Get rid of newline characters
            if(val == '1.0\n'):
                val = '1.00'

            #If method is real, ignore this block
            if(method != "real"):
                if(val >= str(threshold)):
                    #Binary method set similarities to 1, min_threshold keeps value
                    if(method == "binary"):
                       val = '1.00'
                #Below threshold set to 0 for both methods
                else:
                    val = '0.00'
            #Character range from '!' to '}'
            if(char > '00020' and char < '0007e'):
                # Convert string hex to a unicode char.
                key = chr(int(key,16))  
                simvalues[key].append(float(val))
                #print("adding " + key + " as " + str(chr(int(key,16))))
                #print("added " + key + " " + char + " " + val)

def prettyPrintVec(vec):
    """
    Print nicely the similarities with their ASCII char alignment.
    """
    i = 33
    print("[ ", end='')
    for x in vec:
        print(str(chr(i)) + "=" + str(x), end=' ')
        i += 1
    print("]")

def printDict():        
    for value in simvalues:
        print(value, simvalues[value])

def queryDict(unichar):
    if(unichar in simvalues):
        return simvalues[unichar]
    return 0

if __name__ == "__main__":
    #Types of methods are "real", "min_threshold", and "binary"
    valueDict("real",.65)
    #printDict()
    #print(queryDict( chr(int('0003e',16)) ))
    print(prettyPrintVec(queryDict('3')))


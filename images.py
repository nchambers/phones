import numpy as np
from skimage import io
from skimage.transform import resize
from skimage.transform import rotate
import random
import os
import threshParse

# Default path to the unicode images repo.
LOAD_DIR='/home/nchamber/corpora/unicodeDB/unicodeviz/imgs/'
# Grab the environment variable if it exists.
if os.getenv('UNICODEVIZ'):
    LOAD_DIR = os.getenv('UNICODEVIZ')

cache=dict()


def invertOnes(img):
    """
    Convert all values x to (1-x)
    """
    for i in range(0,len(img)):
        for j in range(0,len(img[i])):
            img[i][j][0] = 1.0 - img[i][j][0]

def cleanCorners(img):
    """The image files have small value deviations in the corners of the 34x34 images. 
    This function resets the corners to 1.0 values.
    """
    for i in range(1,3):
        img[i][0] = img[i][33] = 1.0
        img[33-i][0] = img[33-i][33] = 1.0
    #Clean up edge noise from input character files
    img[0] = img[33] = np.ones((34,1))

def noise(img):
    """
    Returns a new image with all non-zero pixel values changed by small amounts.
    """
    
    r = random.randint(0,3)
    if r == 0:   # lighter
        LOW=0.7
        HIGH=1.0
    elif r == 1: # darker
        LOW=1.0
        HIGH=1.3
    else:        # mix lighter+darker
        LOW=0.7
        HIGH=1.3

    N = len(img)
    noise = np.random.uniform(low=LOW, high=HIGH, size=(N,N,1))
    return img*noise

def shimmy(img):
    """
    Shifts all image values in the image matrix +/- 3 rows and cols randomly.
    Values on the edges will wrap to the other side, but most images have
    5-8 empty rows or columns on the edges, so this shouldn't matter.
    """
    dx = random.randint(-3,3)
    dy = random.randint(-3,3)

    if dx >= 0:
        dx += 1
    if dy >= 0:
        dy += 1

    img = np.roll(img, dy, axis=0)
    img = np.roll(img, dx, axis=1)
    return img

def stretch(img):
    """
    Stretches image in both x and y axes, by a random value of 3-5. 
    Returns center of stretched image.
    """
    
    dx = random.randint(3,6)
    dy = random.randint(3,6)

    img = resize(img, (34+2*dx,34+2*dy), mode='constant', cval=0)[dx:34+dx,dy:34+dy]

    return img

def turn(img):
    """
    Rotates image around origin.
    Returns rotated image.
    """

    dr = random.uniform(-20,20)
    if(dr == 0.0):
        dr = 20
    
    img = rotate(img, dr, center=(16,16), mode='constant', cval=0)

    return img

def jiggleImage(img):
    """
    Altering the image with probablistic changes, including turn, stretch, shimmy, and noise.
    """
    # ~16.7% Probability of turning
    if(random.randint(0,6) == 0):
        img = turn(img)

    # 50% Probability of stretching
    if(random.randint(0,2) == 0):
        img = stretch(img)
    
    # 50% Probability of shimmying
    if(random.randint(0,2) == 0):
        img = shimmy(img)

    # 80% Probability of noise
    if(random.randint(0,5) == 0):
        return img

    return noise(img)

#
# Databse of unicode images from:
# https://github.com/PantherLab/v2d-unicodeDB/tree/master/unicode
#
def myReadChar(ch):
    """
    Given a char, finds the file that represents the image of that character.
    Reads into memory and returns a zero-based image.
    """
    #str = repr(ch)
    #print ch, repr(ch), type(repr(ch)), ord(ch[0]), hex(ord(ch[0])), type(hex(ord(ch[0])))

    # Pad the hex with front zeros
    prefixHex = hex(ord(ch[0]))[2:]
    while len(prefixHex) < 5:
        prefixHex = '0' + prefixHex

    # File path to the char image.
    mypath = os.path.join(LOAD_DIR, prefixHex + '.png')

    # If we don't have this char
    if not os.path.exists(mypath):
        print("WARNING: path doesn't exist (", mypath, ") so don't have image of unicode", prefixHex, "from '", ch[0], "'")
        m = np.zeros((34,34))

    else:
        # Read the 34x34 image into a matrix.
        # (flatten is deprecated, but needed if old libraries installed)
        m = io.imread(mypath, as_gray=True, flatten=True)
        if len(m) != 34:
            print("WARNING (size):", mypath, " read size:", len(m), len(m[0]))

    # Expand to nominal 34x34x1 tensor
    m = np.expand_dims(m, -1)
    
    cleanCorners(m)
    invertOnes(m)
 
    return m


def stringToImages(thestring, maxlength, jiggle=False):
    """
    Given a single text string (a full phone number), this will
    return a corresponding 3D image of the characters in
    the string ... the third dimension is just a single value.
    """
    global cache

    # Empty 3d tensor of length maxlength
    imgs = np.empty((maxlength,34,34,1))

    if len(thestring) > maxlength:
        thestring = thestring[0:maxlength]
        
    # Read each char image.
    i = 0
    for x in thestring:
        if not x in cache:
            img = myReadChar(x)
            cache[x] = img
        else:
            img = cache[x]

        if len(img) != 34:
            print("WARNING: read image file of size:", len(img),'x',len(img[0]), 'for ch =', x)
            img = np.zeros((34,34,1))

        if jiggle:
            img = jiggleImage(img)

        imgs[i] = img
        i += 1

    # Pad the end with zeros
    for j in range(i,maxlength):
        imgs[j] = np.zeros((34,34,1))

#    print("THESTRING=%s" % (thestring))
#    for zz in imgs:
#        prettyPrintCharImage(zz)
        
    return imgs

def repeatLabels(labels, repeat):
    """Expects labels tensor: NxMxC where N=#datums, M=#chars, C=#categories
    """
    if repeat == 0: return labels

    newlabels = np.empty( (len(labels)*(repeat+1), len(labels[0]), len(labels[0][0])) )
    i = 0
    for labeling in labels:
        for r in range(0,repeat+1):
            newlabels[i] = labeling
            i += 1
    return newlabels

def repeatGoldDigits(labels, repeat):
    """Expects labels vector of size N, with N phone numbers as integers.
    """
    if repeat == 0: return labels

    newlabels = np.empty( (len(labels)*(repeat+1)) )   # only diff line from repeatLabels
    i = 0
    for phone in labels:
        for r in range(0,repeat+1):
            newlabels[i] = phone
            i += 1
    return newlabels

def charsToImages(texts, max_sequence_length, jiggle=False, repeat=0):
    """
    Given a list of text strings (full phone numbers), this will
    return a corresponding list of 3-d images of the characters in
    each text string ... the third dimension is just a single value.

    Returns a list of mask values as well, which is just an array of
    1's until the number of text chars runs out, and then it is 0's
    until the end of max_sequence_length. Can be used in a model to
    multiply out ending pads.
    """
    # Sanity check
    if repeat > 0 and not jiggle:
        print("ERROR charsToImages(): can't repeat>0 if jiggle=False")
        sys.exit(1)

    print("charsToImages with " + str(len(texts)) + " texts and jiggle=" + str(jiggle))

    # data = np.empty((len(texts),max_sequence_length,34,34,1))
    # maskvals = np.empty((len(texts),max_sequence_length,1))
    # for i in range(0,len(texts)):
    #     obscured = texts[i]
    #     imgs = stringToImages(obscured, max_sequence_length, jiggle)
    #     data[i] = imgs
    #     if i % 100 == 0:
    #         print(str(i) + "...")

    #     mask = np.append( np.ones(min(max_sequence_length,len(obscured))), np.zeros(max(0,max_sequence_length-len(obscured))) )
    #     mask = np.expand_dims(mask, axis=1)
    #     maskvals[i] = mask

    data = np.empty((len(texts)*(repeat+1),max_sequence_length,34,34,1))
    maskvals = np.empty((len(texts)*(repeat+1),max_sequence_length,1))
    i = 0
    for obscured in texts:
        # Generate multiple versions if we're jiggling the images and repeat > 0
        for r in range(0,repeat+1):
            imgs = stringToImages(obscured, max_sequence_length, jiggle)
            data[i] = imgs
            if i % 100*(repeat+1) == 0:
                print(str(i) + "...")

            mask = np.append( np.ones(min(max_sequence_length,len(obscured))), np.zeros(max(0,max_sequence_length-len(obscured))) )
            mask = np.expand_dims(mask, axis=1)
            maskvals[i] = mask
            
            # print("obscured i=%d r=%d = %s" % (i,r,obscured))
            # print("mask " + str(mask))
            # x = 0
            # for img in data[i]:
            #     prettyPrintCharImage(img)
            #     print()
            #     x += 1
            #     if x == len(obscured): break

            i += 1

#    print(texts[0])
#    print(maskvals[0])
    print("charsToImages() returning %d datums" % (len(data)))
    return (data, maskvals)


def prettyPrintCharImage(img):
    #print("len=%d" % (len(img)))
    for i in range(0,len(img)):
        #print("len i=%d =%d" % (i, len(img[i])))
        for j in range(0,len(img[i])):
            if img[i][j][0] == 1.0:
                print(" 1 ", end=' ')
            elif img[i][j][0] < 0.05:
                print(" 0 ", end=' ')
            else:
                print("%.1f" % (img[i][j][0]), end=' ')
            #print(img[i][j][0], end=' ')
        print()
    print()

# DEBUGGING INFO ABOUT CHAR IMAGE FILES
def statsAllImages():
    n = 0
    rows = [0] * 34

    for f in os.listdir(LOAD_DIR):
        mypath = os.path.join(LOAD_DIR, f)
        m = io.imread(mypath, as_gray=True, flatten=True)

        # Expand to nominal 34x34x1 tensor
        img = np.expand_dims(m, -1)
        cleanCorners(img)

        # Check each row for pixel activation.
        if len(img) == 34:

            # SANITY CHECK: some images are filled with 0.9 values...
            pointnines = 0
            for i in range(0,len(img)):
                val = img[0][i]
                if val > 0.85 and val < 0.95:
                    pointnines += 1
            if pointnines > 30:
                continue

            # NORMAL.
            for i in range(0,len(img)):
                for j in range(0,len(img[i])):
                    val = img[i][j]

                    if val < 0.94:
                        rows[i] += 1
                        #prettyPrintCharImage(img)
                        break
            n = n + 1
        else:
            print("Bad char at: " + mypath)
            print(mypath, " read size:", len(m), len(m[0]))
    # Stats!
    print("%d rows" % (n))
    for i in range(0,len(rows)):
        print("row %d had %d non-zeros" % (i, rows[i]))


def stringToCharSims(thestring, maxlength):
    """
    Given a single text string (a full phone number), this will
    return a corresponding vector of each character in
    the string ... vector is the similarity scores w/ ASCII chars.
    """
    # Length of the similarity vector.
    N = 93

    # Empty 3d tensor of length maxlength
    sims = np.zeros((maxlength,N))

    if len(thestring) > maxlength:
        thestring = thestring[0:maxlength]
        
    # Lookup each char's similarity vector
    i = maxlength-len(thestring)
    for x in thestring:
        #print("Looking up " + x + " from " + thestring)
        vec = threshParse.queryDict(x)

        if vec == 0:
            vec = np.zeros((N,))

        if len(vec) != N:
            print("WARNING: sim vector of size:", len(vec), 'for ch =', x)
            vec = np.zeros((N,))

        #threshParse.prettyPrintVec(vec)

        sims[i] = vec
        i += 1

    return sims


def charsToCharSims(texts, max_sequence_length, repeat=0):
    """
    Given a list of text strings (full phone numbers), this will
    return a corresponding list of sim vectors of the characters in
    each text string. The vector is how similar each char is to all
    the ASCII chars.
    """
    print("charsToCharSims with " + str(len(texts)) + " texts")

    # Check if the database is loaded.
    if not threshParse.simvalues:
        print("images.py loading char similarities data from file...")
        threshParse.valueDict('real', 0.0)
    vecN = 93  # length of similarity vector that is returned

    #test = threshParse.queryDict('0ac52')  # random unicode char

    data = np.empty((len(texts)*(repeat+1),max_sequence_length,vecN))
    i = 0
    for obscured in texts:
        # Generate multiple versions if we're jiggling the images and repeat > 0
        for r in range(0,repeat+1):
            vecs = stringToCharSims(obscured, max_sequence_length)
            data[i] = vecs
            if i % 100*(repeat+1) == 0:
                print(str(i) + "...")
            i += 1

    print("charsToCharSims() returning %d datums" % (len(data)))
    return data

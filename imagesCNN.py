from keras.layers import Dense, Input, Flatten, Conv2D, MaxPooling2D, Dropout, TimeDistributed
from keras import Sequential
from keras.layers.normalization import BatchNormalization

#
# Two CNNs stacked. 
# If you want just one CNN, send in NUMFILTERS2=0
# @return A pair: (1) Input layer, and (2) TimeDistributed layer
#
# Creates Keras layers from the input (34x34 character images) to the CNN
# layers to the output as a TimeDistributed layer of embeddings, one embedding
# per character (OUTPUTDIM sized).
#
def createCNN(MAXLENGTH, NUMFILTERS1, NUMFILTERS2, NUMFILTERS3, OUTPUTDIM):

    # INPUT shape
    sequence_input = Input(shape=(MAXLENGTH,34,34,1), dtype='float32')

    # SINGLE IMAGE (keras sequential add)
    cnn = Sequential()
    temp = Conv2D(NUMFILTERS1, (3,3), activation='relu', padding='same', input_shape=(34,34,1), name="conv2d-1")
    cnn.add(temp)

    print("CNN out shape:",temp.name,temp.output_shape)

    # Mimicking part of setup in https://yashk2810.github.io/Applying-Convolutional-Neural-Network-on-the-MNIST-dataset/
    # Great visual of stacked-CNN shapes: https://stackoverflow.com/questions/54098364/understanding-channel-in-convolution-neural-network-cnn-input-shape-and-output
    cnn.add(BatchNormalization(axis=-1))
    if NUMFILTERS2 > 0:
        cnn.add(Conv2D(NUMFILTERS2, (3,3), activation='relu', padding='same', name="conv2d-2"))
        cnn.add(BatchNormalization(axis=-1))
    if NUMFILTERS3 > 0:
        cnn.add(Conv2D(NUMFILTERS3, (3,3), activation='relu', padding='same', name="conv2d-3"))
        cnn.add(BatchNormalization(axis=-1))

    cnn.add(MaxPooling2D(pool_size=(2,2)))
    cnn.add(Flatten())
    cnn.add(Dense(OUTPUTDIM, activation='softmax'))        # convert to an embedding   TODO: activation None? "linear layer"
#    cnn.add(Dense(OUTPUTDIM))        # No activation, just linear
#    cnn.add(Dense(OUTPUTDIM, activation='relu'))        # convert to an embedding   TODO: activation None? "linear layer"
#    cnn.add(Dense(OUTPUTDIM, activation='relu'))        # convert to an embedding   TODO: activation None? "linear layer"

    # Wrap single image processing with a time distributed object.
    l_td = TimeDistributed(cnn)(sequence_input)  # Input is N images of 34x34 size

    # Debug output of parameter sizes.
    cnn.summary()
    return (sequence_input, l_td)

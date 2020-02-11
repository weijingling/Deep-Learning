# -*- coding: utf-8 -*-
"""
@Time    : Tue Feb 11 10:34:32 2020
@Author  : Jing-Ling, Wei
@Software: Spyder (Python 3.7)
"""
from keras.models import Sequential # Activate NN
from keras.layers import Convolution2D # Convolution Operation
from keras.layers import Input
from keras.layers import Dense # Fully Connected Networks
from keras.models import Model
from keras.datasets import mnist
from matplotlib import pyplot as plt
from keras.utils import np_utils

class ConvolutionVsFC():
    def __init__(self, x_train, x_test, y_train, y_test):
        self.X_TRAIN = x_train
        self.X_TEST = x_test
        self.Y_TRAIN = y_train
        self.Y_TEST = y_test
        
        self.CNN_MODEL = 0
        self.FC_MODEL = 0
        
        self.load_data()
        self.preprocess_input_data()
        self.preprocess_output_data()
        self.build_cnn_model()
        self.build_fc_model()
    
    def load_data(self):
        # print(self.X_TRAIN.shape) # (60000, 28, 28)
        
        # Plotting first sample of X_TRAIN
        plt.imshow(self.X_TRAIN[0])
    
    def preprocess_input_data(self):
        # When using the Theano backend, you must explicitly declare a dimension for the depth of the input image.
        # For example, a full-color image with all 3 RGB channels will have a depth of 3.
        # MNIST images only have a depth of 1, but we must explicitly declare that.
        # Reshape input data
        self.X_TRAIN = self.X_TRAIN.reshape(self.X_TRAIN.shape[0], 1, 28, 28)
        self.X_TEST = self.X_TEST.reshape(self.X_TEST.shape[0], 1, 28, 28)
        # print(self.X_TRAIN.shape) # (60000, 1, 28, 28)
        
        # Convert our data type to float32 and normalize our data values to the range [0, 1].
        self.X_TRAIN = self.X_TRAIN.astype('float32')
        self.X_TEST = self.X_TEST.astype('float32')
        self.X_TRAIN /= 255
        self.X_TEST /= 255
    
    def preprocess_output_data(self):
        # print(self.Y_TRAIN.shape) # (60000,)
        
        # We have 10 different classes, one for each digit, but we only have a 1-dimensional array.
        # print(self.Y_TRAIN[:10]) # [5 0 4 1 9 2 1 3 1 4]
        
        # The y_train and y_test data are not split into 10 distinct class labels,
        # but rather are represented as a single array with the class values.
        # Convert 1-dimensional class arrays to 10-dimensional class matrices
        self.Y_TRAIN = np_utils.to_categorical(self.Y_TRAIN, 10)
        self.Y_TEST = np_utils.to_categorical(self.Y_TEST, 10)
        # print(self.Y_TRAIN.shape) # (60000, 10)
        
    def build_cnn_model(self):        
        # Declare sequential model
        self.CNN_MODEL = Sequential()
        
        # Declare the input layer
        # Kernel Numbers=32, Kernel Size=3*3, Input Shape=28*28*1
        # The first 3 parameters correspond to the number of convolution filters to use, the number of rows in each convolution kernel, and the number of columns in each convolution kernel, respectively.
        # *Note: The step size is (1,1) by default, and it can be tuned using the 'subsample' parameter.
        # The input shape parameter should be the shape of 1 sample.
        # It's the same (28, 28, 1) that corresponds to the (height, width, depth) of each digit image.
        self.CNN_MODEL.add(Convolution2D(filters=32, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='relu'))
        # Show model structure
        print(self.CNN_MODEL.summary()) # Output Shape=(None, 26, 26, 32), Parameter Number=320 # Parameter Number=(Kernel_H*W*channels + 1) * Kernel numbers

    def build_fc_model(self):
        # Declare fully connect model
        self.FC_MODEL = Sequential()
        
        i = Input(shape=(784,)) # Flatten 28*28*1
        o = Dense(288)(i) # Neuron number=(3*3*1)*32
        
        self.FC_MODEL = Model(inputs=i, outputs=o)
        
        # Show model structure
        print(self.FC_MODEL.summary()) # Output Shape=(None, 784) & (None, 288), Parameter Number=226,080 # Parameter Number=Input*output + output
    
if __name__ == '__main__':
    # Load pre-shuffled MNIST data into train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    ConvolutionVsFC(x_train, x_test, y_train, y_test)
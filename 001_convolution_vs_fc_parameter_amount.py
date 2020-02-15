# -*- coding: utf-8 -*-
"""
@Time    : Tue Feb 11 10:34:32 2020
@Author  : Jing-Ling, Wei
@Software: Spyder (Python 3.7)
"""
from keras.models import Sequential # Activate NN
from keras.layers import Convolution2D # Convolution Operation
from keras.layers import Dense # Fully Connected Networks
from keras.datasets import mnist
from matplotlib import pyplot as plt
from keras.utils import np_utils

class ConvolutionVsFC():
    def __init__(self, train_data, test_data, train_label, test_label):
        self.TRAIN_DATA = train_data
        self.TEST_DATA = test_data
        self.TRAIN_LABEL = train_label
        self.TEST_LABEL = test_label
        
        self.CNN_MODEL = 0
        self.FC_MODEL = 0
        
        self.load_data()
        self.preprocess_input_data()
        self.preprocess_output_data()
        self.build_cnn_model()
        self.build_fc_model()
    
    def load_data(self):
#        print(self.TRAIN_DATA.shape) # (60000, 28, 28)
        
        # =============================================================================
        #         Plotting first sample of TRAIN_DATA
        # =============================================================================
        plt.imshow(self.TRAIN_DATA[0])
    
    def preprocess_input_data(self):
        # =============================================================================
        #         When using the Theano backend, you must explicitly declare a dimension for the depth of the input image.
        #         For example, a full-color image with all 3 RGB channels will have a depth of 3.
        #         MNIST images only have a depth of 1, but we must explicitly declare that.
        # =============================================================================
        # =============================================================================
        #         Reshape input data
        # =============================================================================
        self.TRAIN_DATA = self.TRAIN_DATA.reshape(self.TRAIN_DATA.shape[0], 1, 28, 28)
        self.TEST_DATA = self.TEST_DATA.reshape(self.TEST_DATA.shape[0], 1, 28, 28)
#        print(self.TRAIN_DATA.shape) # (60000, 1, 28, 28)
        
        # =============================================================================
        #         Convert the data type to float32 and normalize the data values to the range [0, 1].
        # =============================================================================
        self.TRAIN_DATA = self.TRAIN_DATA.astype('float32')
        self.TEST_DATA = self.TEST_DATA.astype('float32')
        self.TRAIN_DATA /= 255
        self.TEST_DATA /= 255
    
    def preprocess_output_data(self):
#        print(self.TRAIN_LABEL.shape) # (60000,)
        
        # =============================================================================
        #         There are 10 different classes, one for each digit, but only a 1-dimensional array.
        # =============================================================================
#        print(self.TRAIN_LABEL[:10]) # [5 0 4 1 9 2 1 3 1 4]
        
        # =============================================================================
        #         The TRAIN_LABEL and TEST_LABEL data are not split into 10 distinct class labels,
        #         but rather are represented as a single array with the class values.
        # =============================================================================
        # =============================================================================
        #         Convert 1-dimensional class arrays to 10-dimensional class matrices
        # =============================================================================
        self.TRAIN_LABEL = np_utils.to_categorical(self.TRAIN_LABEL, 10)
        self.TEST_LABEL = np_utils.to_categorical(self.TEST_LABEL, 10)
#        print(self.TRAIN_LABEL.shape) # (60000, 10)
        
    def build_cnn_model(self):        
        # =============================================================================
        #         Declare sequential model
        # =============================================================================
        self.CNN_MODEL = Sequential()
        
        # =============================================================================
        #         Declare the input layer
        # =============================================================================
        # =============================================================================
        #         Kernel Numbers=32, Kernel Size=3*3, Input Shape=28*28*1
        #         The first 3 parameters correspond to the number of convolution filters to use, 
        #         the number of rows in each convolution kernel, and the number of columns in each convolution kernel, respectively.
        #         *Note: The step size is (1,1) by default, and it can be tuned using the 'subsample' parameter.
        #         The input shape parameter should be the shape of 1 sample.
        #         It's the same (28, 28, 1) that corresponds to the (height, width, depth) of each digit image.
        # =============================================================================
        self.CNN_MODEL.add(Convolution2D(filters=32, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='relu'))
        
        # =============================================================================
        #         Show model structure
        # =============================================================================
        print(self.CNN_MODEL.summary()) # Output Shape=(None, 26, 26, 32), Parameter Number=320 # Parameter Number=(Kernel_H*W*channels + 1) * Kernel numbers

    def build_fc_model(self):
        # =============================================================================
        #         Declare fully connect model
        # =============================================================================
        self.FC_MODEL = Sequential()
        self.FC_MODEL.add(Dense(288, input_dim=784))
        
        # =============================================================================
        #         Show model structure
        # =============================================================================
        print(self.FC_MODEL.summary()) # Output Shape=(None, 288), Parameter Number=226,080 # Parameter Number=Input*output + output
    
if __name__ == '__main__':
    # =============================================================================
    #     Load pre-shuffled MNIST data into train and test sets
    # =============================================================================
    (train_data, train_label), (test_data, test_label) = mnist.load_data()
    
    ConvolutionVsFC(train_data, test_data, train_label, test_label)
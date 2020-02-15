# -*- coding: utf-8 -*-
"""
@Time    : Fri Feb 14 11:35:21 2020
@Author  : Jing-Ling, Wei
@Software: Spyder (Python 3.7)
"""

from keras.models import Sequential # Activate NN
from keras.layers import Convolution2D # Convolution Operation
from keras.layers import BatchNormalization
from keras.layers import Activation

class ConvolutionBatchNormalization():
    def __init__(self):
        self.build_model_batch_normalization()
        
    def build_model_batch_normalization(self):        
        MODEL = Sequential()
        
        # =============================================================================
        #         BatchNormalization:
        #         momentum: Momentum for the moving mean and the moving variance.
        #         epsilon: Small float added to variance to avoid dividing by zero.
        # =============================================================================
        
        MODEL.add(Convolution2D(filters=32, kernel_size=(3, 3), input_shape=(32, 32, 3), padding='same'))
        # Output Shape = (None, 32, 32, 32), Param# = 896
        MODEL.add(BatchNormalization()) 
        # Output Shape = (None, 32, 32, 32), Param# = 128
        MODEL.add(Activation('sigmoid'))
        # Output Shape = (None, 32, 32, 32), Param# = 0
        MODEL.add(Convolution2D(filters=32, kernel_size=(3, 3), padding='same'))
        # Output Shape = (None, 32, 32, 32), Param# = 9248
        MODEL.add(BatchNormalization()) 
        # Output Shape = (None, 32, 32, 32), Param# = 128
        MODEL.add(Activation('relu'))
        # Output Shape = (None, 32, 32, 32), Param# = 0
        
        print(MODEL.summary()) # Total params = 10,400
        
if __name__ == '__main__':    
    ConvolutionBatchNormalization()
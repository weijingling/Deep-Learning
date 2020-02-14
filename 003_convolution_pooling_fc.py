# -*- coding: utf-8 -*-
"""
@Time    : Thu Feb 13 15:01:41 2020
@Author  : Jing-Ling, Wei
@Software: Spyder (Python 3.7)
"""

from keras.models import Sequential # Activate NN
from keras.layers import Convolution2D # Convolution Operation
from keras.layers import MaxPooling2D # Pooling
from keras.layers import Flatten
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense # Fully Connected Networks

class ConvolutionPoolingFC():
    def __init__(self):
        self.build_model_pooling_flatten()
        self.build_model_pooling()
        
    def build_model_pooling_flatten(self):        
        MODEL = Sequential()
        
        MODEL.add(Convolution2D(filters=32, kernel_size=(3, 3), padding='same',input_shape=(32, 32, 3)))
        # Output Shape = (None, 32, 32, 32), Param# = 896
        MODEL.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # Output Shape = (None, 16, 16, 32), Param# = 0
        
        MODEL.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
        # Output Shape = (None, 16, 16, 64), Param# = 18496
        MODEL.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # Output Shape = (None, 8, 8, 64), Param# = 0
        
        MODEL.add(Convolution2D(filters=128, kernel_size=(3, 3), padding='same'))
        # Output Shape = (None, 8, 8, 128), Param# = 73856
        MODEL.add(MaxPooling2D(pool_size=(1, 1), strides=(1, 1)))
        # Output Shape = (None, 8, 8, 128), Param# = 0
        
        MODEL.add(Convolution2D(filters=10, kernel_size=(3, 3), padding='same'))
        # Output Shape = (None, 8, 8, 10), Param# = 11530
        MODEL.add(Flatten())
        # Output Shape = (None, 640), Param# = 0
        
        MODEL.add(Dense(28))
        # Output Shape = (None, 28), Param# = 17948
        
        print(MODEL.summary()) # Total params = 122,726
        
    def build_model_pooling(self):        
        MODEL = Sequential()
        
        MODEL.add(Convolution2D(filters=32, kernel_size=(3, 3), padding='same',input_shape=(32, 32, 3)))
        # Output Shape = (None, 32, 32, 32), Param# = 896
        MODEL.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # Output Shape = (None, 16, 16, 32), Param# = 0
        
        MODEL.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
        # Output Shape = (None, 16, 16, 64), Param# = 18496
        MODEL.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # Output Shape = (None, 8, 8, 64), Param# = 0
        
        MODEL.add(Convolution2D(filters=128, kernel_size=(3, 3), padding='same'))
        # Output Shape = (None, 8, 8, 128), Param# = 73856
        MODEL.add(MaxPooling2D(pool_size=(1, 1), strides=(1, 1)))
        # Output Shape = (None, 8, 8, 128), Param# = 0
        
        MODEL.add(Convolution2D(filters=10, kernel_size=(3, 3), padding='same'))
        # Output Shape = (None, 8, 8, 10), Param# = 11530
        MODEL.add(GlobalAveragePooling2D())
        # Output Shape = (None, 10), Param# = 0
        
        MODEL.add(Dense(28))
        # Output Shape = (None, 28), Param# = 308
        
        print(MODEL.summary()) # Total params = 105,086
        
if __name__ == '__main__':    
    ConvolutionPoolingFC()
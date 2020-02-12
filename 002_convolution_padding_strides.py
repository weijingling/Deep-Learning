# -*- coding: utf-8 -*-
"""
@Time    : Wed Feb 12 17:05:46 2020
@Author  : Jing-Ling, Wei
@Software: Spyder (Python 3.7)
"""
from keras.models import Sequential # Activate NN
from keras.layers import Convolution2D # Convolution Operation
from keras.layers import Input
from keras.layers import Dense # Fully Connected Networks
from keras.models import Model

class ConvolutionPaddingStrides():
    def __init__(self):
        self.FILTERS = 32
        self.KERNEL_SIZE = (6, 6)
        self.INPUT_SHAPE = (13, 13, 1)
        
        self.set_paddingsame_stride1()
        self.set_paddingsame_stride2()
        self.set_paddingvalid_stride1()
        self.set_paddingvalid_stride2()
        
    def set_paddingsame_stride1(self):
        CNN_MODEL = Sequential()
        # strides=(1, 1), padding='same'
        CNN_MODEL.add(Convolution2D(filters=self.FILTERS, kernel_size=self.KERNEL_SIZE, input_shape=self.INPUT_SHAPE, strides=(1,1), padding='same'))

        print("--- strides=(1, 1), padding=same ---")
        print(CNN_MODEL.summary()) # Output Shape = (None, 13, 13, 32)
        
    def set_paddingsame_stride2(self):
        CNN_MODEL = Sequential()
        # strides=(2, 2), padding='same'
        CNN_MODEL.add(Convolution2D(filters=self.FILTERS, kernel_size=self.KERNEL_SIZE, input_shape=self.INPUT_SHAPE, strides=(2,2), padding='same'))

        print("--- strides=(2, 2), padding=same ---")
        print(CNN_MODEL.summary()) # Output Shape = (None, 7, 7, 32)
        
    def set_paddingvalid_stride1(self):
        CNN_MODEL = Sequential()
        # strides=(1, 1), padding='valid'
        CNN_MODEL.add(Convolution2D(filters=self.FILTERS, kernel_size=self.KERNEL_SIZE, input_shape=self.INPUT_SHAPE, strides=(1,1), padding='valid'))

        print("--- strides=(1, 1), padding=valid ---")
        print(CNN_MODEL.summary()) # Output Shape = (None, 8, 8, 32)
        
    def set_paddingvalid_stride2(self):
        CNN_MODEL = Sequential()
        # strides=(2, 2), padding='valid'
        CNN_MODEL.add(Convolution2D(filters=self.FILTERS, kernel_size=self.KERNEL_SIZE, input_shape=self.INPUT_SHAPE, strides=(2,2), padding='valid'))

        print("--- strides=(2, 2), padding=valid ---")
        print(CNN_MODEL.summary()) # Output Shape = (None, 4, 4, 32)
        
if __name__ == '__main__':
    
    ConvolutionPaddingStrides()
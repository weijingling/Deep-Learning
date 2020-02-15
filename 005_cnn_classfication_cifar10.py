# -*- coding: utf-8 -*-
"""
@Time    : Sat Feb 15 08:10:39 2020
@Author  : Jing-Ling, Wei
@Software: Spyder (Python 3.7)
"""
from keras.datasets import cifar10
from keras.models import Sequential # Activate NN
from keras.layers import Convolution2D # Convolution Operation
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense # Fully Connected Networks
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def show_train_log(train_log):
    fig = plt.gcf()
    fig.set_size_inches(16, 6)
    plt.subplot(121)
    plt.plot(train_log.history["acc"])
    plt.plot(train_log.history["val_acc"])
    plt.title("Train History")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["train", "validation"], loc="upper left")
    plt.subplot(122)
    plt.plot(train_log.history["loss"])
    plt.plot(train_log.history["val_loss"])
    plt.title("Train History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()

class CnnClassificationModel():
    def __init__(self, train_data, test_data, train_label, test_label):
        self.TRAIN_DATA = train_data
        self.TEST_DATA = test_data
        self.TRAIN_LABEL = train_label
        self.TEST_LABEL = test_label
        
        self.TRAIN_DATA_NORMALIZE = 0
        self.TEST_DATA_NORMALIZE = 0
        self.TRAIN_LABEL_ONEHOT = 0
        self.TEST_LABEL_ONEHOT = 0
        
        self.CNN_MODEL = 0
        
        self.load_data()
        self.preprocess_input_data()
        self.preprocess_output_data()
        self.build_cnn_model()
        self.train_cnn_model()
        self.evaluate_cnn_model()
        self.predict_cnn_model()
    
    def load_data(self):
        print(self.TRAIN_DATA.shape) # (50000, 32, 32, 3)
        print(self.TEST_DATA.shape) # (10000, 32, 32, 3)
        print(self.TRAIN_LABEL.shape) # (50000, 1)
        print(self.TEST_LABEL.shape) # (10000, 1)
        
        # =============================================================================
        #         Plotting first sample of TRAIN
        # =============================================================================
        plt.imshow(self.TRAIN_DATA[0])
        print(self.TRAIN_LABEL[0]) # [6]
    
    def preprocess_input_data(self):        
        # =============================================================================
        #         Convert the data type to float32 and normalize the data values to the range [0, 1].
        # =============================================================================
        self.TRAIN_DATA_NORMALIZE = self.TRAIN_DATA.astype('float32')
        self.TEST_DATA_NORMALIZE = self.TEST_DATA.astype('float32')
        self.TRAIN_DATA_NORMALIZE /= 255
        self.TEST_DATA_NORMALIZE /= 255
    
    def preprocess_output_data(self):
        # =============================================================================
        #         Convert 1-dimensional class arrays to 10-dimensional class matrices
        # =============================================================================
        self.TRAIN_LABEL_ONEHOT = np_utils.to_categorical(self.TRAIN_LABEL, 10)
        self.TEST_LABEL_ONEHOT = np_utils.to_categorical(self.TEST_LABEL, 10)
        
        print(self.TRAIN_LABEL_ONEHOT.shape) # (50000, 10)
        print(self.TEST_LABEL_ONEHOT.shape) # (10000, 10)
        
    def build_cnn_model(self):
        self.CNN_MODEL = Sequential()
        
        self.CNN_MODEL.add(Convolution2D(filters=32, kernel_size=(3, 3), input_shape=(32, 32, 3), padding='same', activation='relu'))
        # Output Shape = (None, 32, 32, 32), Param# = 896
        self.CNN_MODEL.add(Dropout(0.25))
        # Output Shape = (None, 32, 32, 32), Param# = 0
        self.CNN_MODEL.add(MaxPooling2D(pool_size=(2, 2)))
        # Output Shape = (None, 16, 16, 32), Param# = 0
        
        self.CNN_MODEL.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
        # Output Shape = (None, 16, 16, 64), Param# = 18496
        self.CNN_MODEL.add(Dropout(0.25))
        # Output Shape = (None, 16, 16, 64), Param# = 0
        self.CNN_MODEL.add(MaxPooling2D(pool_size=(2, 2)))
        # Output Shape = (None, 8, 8, 64), Param# = 0
        
        self.CNN_MODEL.add(Flatten())
        # Output Shape = (None, 4096), Param# = 0
        self.CNN_MODEL.add(Dropout(0.25))  
        # Output Shape = (None, 4096), Param# = 0
        
        self.CNN_MODEL.add(Dense(1024, activation='relu'))
        # Output Shape = (None, 1024), Param# = 4195328
        self.CNN_MODEL.add(Dropout(0.25))  
        # Output Shape = (None, 1024), Param# = 0
        
        self.CNN_MODEL.add(Dense(10, activation='softmax'))
        # Output Shape = (None, 10), Param# = 10250
        
        print(self.CNN_MODEL.summary()) # Param# = 4224970 
    
    def train_cnn_model(self):        
        self.CNN_MODEL.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        train_log = self.CNN_MODEL.fit(self.TRAIN_DATA_NORMALIZE, self.TRAIN_LABEL_ONEHOT, validation_split=0.2, epochs=20, batch_size=128, verbose=2)
        
        show_train_log(train_log)
    
    def evaluate_cnn_model(self):
        scores = self.CNN_MODEL.evaluate(self.TEST_DATA_NORMALIZE, self.TEST_LABEL_ONEHOT)
        print("Accuracy = ", scores)  
        print("Accuracy = ", scores[1])
        
    def predict_cnn_model(self):
        prediction = self.CNN_MODEL.predict_classes(self.TEST_DATA_NORMALIZE)
        print(prediction)
        print(prediction[:10])
        print(self.TEST_LABEL[:10])
        
        count = 0
        for i in range(len(prediction)):
            if(np.argmax(prediction[i]) == np.argmax(self.TEST_LABEL[i])):
                count += 1
        score = count/len(prediction)
        print('Accuracy = %.2f %s' % (score*100, '%'))
                
if __name__ == '__main__':
    # =============================================================================
    #     Load pre-shuffled Cifar10 data into train and test sets
    # =============================================================================
    (train_data, train_label), (test_data, test_label) = cifar10.load_data()
    
    CnnClassificationModel(train_data, test_data, train_label, test_label)
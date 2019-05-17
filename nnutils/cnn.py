#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import os
import warnings
import math
import sys
import time
import numpy as np
import keras
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout
from keras.models import load_model, Sequential
from keras.utils import plot_model
import tensorflow as tf
import matplotlib as mpl


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide messy TensorFlow warnings
warnings.filterwarnings("ignore")  # Hide messy Numpy warnings

mpl.use('TkAgg')  # or whatever other backend that you want
if mpl:
    import matplotlib.pyplot as plt

##########################################################################
##########################################################################
##########################################################################
''' MODEL '''


# def buildCNN(....):
#     pass
# model =  ...

# return model


##########################################################################
##########################################################################
##########################################################################
''' PREDICTION '''


class CNN(object):

    def __init__(self, model_h5_file_path, seq_len, bias_window_size=10,  num_input_signals=61):
        self.data = CNNOnlineData(seq_len)
        self.pred_history = np.zeros(bias_window_size)
        self.bias_window_size = bias_window_size
        self.init_ct = bias_window_size
        self.model = None
        self.reg_name = model_h5_file_path.split("/")[-1]
        try:
            self.model = load_model(model_h5_file_path)
            self.model.predict(np.zeros((1, seq_len, num_input_signals)))  # void prediction to create the graph (multithread problem in tensorflow with GPU)
            # self.model._make_predict_function() # https://github.com/fchollet/keras/issues/2397, https://github.com/fchollet/keras/issues/6124
            # self.graph = tf.get_default_graph()
            print("Regressors found and loaded!\n" + model_h5_file_path)
        except Exception as e:
            print("None regressor found!\n " + model_h5_file_path)
            print(e)

    def update(self, msg):
        pass

    def predict(self):
        pass
        # pred = self.model.predict( )
        # return pred

    def reset(self):
        pass

    def isReady(self):
        pass

    def getMean(self):
        pass


##########################################################################
def testCNN(model, xTest, yTest, name, plotsetting=None):
    ''' @model can be either the .h5 model or the path (str) to the model'''
    pass
    # yPred = model.predict(xTest)
    # return yPred


##########################################################################
##########################################################################
##########################################################################
''' data sequence for training '''


class CNNOfflineData(object):
    def __init__(self, seq_len, overlap, normalise=False, number_input_signals=None):
        self.seq_len = seq_len
        self.overlap=overlap
        self.normalise = normalise
        self.number_input_signals = number_input_signals


    def prepareData(self, x, y, axis=None):
        s=int((1-self.overlap)*self.seq_len)
        i=0
        n=0
        X=list()
        Y=list()
        while i+self.seq_len<=len(x):
            X.append(x[i:(i+self.seq_len)])
            Y.append(y[i+self.seq_len]) #as output, for each of 6 finger i put the last value of that window
            i=i+s
            n=n+1
        return X, Y

    def normalise_windows(self, window_data_list):
        pass
        # normalised_window_data_list = []
        # return normalised_window_data_list


##########################################################################
##########################################################################
##########################################################################
''' data stream for prediction '''


class CNNOnlineData(object):

    def __init__(self, seq_len):
        pass

    def update(self, msg):
        pass

    def get(self):
        pass
        # return w

#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import keras
import os
import numpy as np
from scipy.io import loadmat, savemat
import matplotlib as mpl

from nnutils.rnn import RNNOfflineData, buildRNN

mpl.use('TkAgg')  # or whatever other backend that you want
if mpl:
    import matplotlib.pyplot as plt


##########################################################################
##########################################################################
##########################################################################

# ⬢⬢⬢⬢⬢➤ DATA

number_input_signals = 8
number_output_signals = 15
seq_len = 10
plot_input_data_flg = False

# ➤➤➤➤➤ Extract DATA
# input
# xdata_path = "/home/riccardo/tests/test_emg/data/emg.mat"
xtraindata_path = "/home/riccardo/tests/test_emg/data/xtrain.mat"
xtraindata = loadmat(xtraindata_path)
xTrain = xtraindata['train']  # emg

# output
# ydata_path = "/home/riccardo/tests/test_emg/data/leap.mat"
ytraindata_path = "/home/riccardo/tests/test_emg/data/ytrain.mat"
ytraindata = loadmat(ytraindata_path)
yTrain = ytraindata['train']  # joints


# ➤➤➤➤➤ Prepare DATA
rnndata = RNNOfflineData(seq_len, normalise=False, number_input_signals=number_input_signals)
xTrain, yTrain = rnndata.prepareData(xTrain, yTrain)
print("xTrain = {}\nyTrain ={} \n\n\n".format(xTrain.shape, yTrain.shape))

# ➤➤➤➤➤ Plot DATA
if plot_input_data_flg:
    plt.figure()
    for i in range(seq_len):
        plt.plot(np.reshape(xTrain[:, i, :], (xTrain.shape[0], 8)), label="xTrain")
    plt.figure()
    # plt.plot(yTrain, label="yTrain")
    plt.legend()
    plt.show()

##########################################################################
##########################################################################
##########################################################################

# ⬢⬢⬢⬢⬢➤ TRAINING

# PARAMETRS
save_model_flg = False
save_model_path = "/home/riccardo/MEGA/test_ml/emg"
model_name = "pippo"

lstm_cells = [100, 100, 100, 100, 100]
dropout_value = [0.5, 0.5, 0.5, 0.5, 0.5]
dense_neurons = [16, 16, 16,  number_output_signals]

epochs = 12
learning_rate = 1e-3
rho = 0.9  # decay factor average over the square of the gradients
decay = 0  # decays the learning rate over time, so we can move even closer to the local minimum in the end of training
validation_split = 0.05

# ⬢⬢⬢⬢⬢➤ MODEL

# BUILD
rnnmodel = buildRNN(
    number_input_signals=number_input_signals,
    number_output_signals=number_output_signals,
    seq_len=seq_len,
    dropout_value=dropout_value,
    lstm_cells=lstm_cells,
    dense_activation=None,
    dense_neurons=dense_neurons
)

# OPTIMIZATION SETTINGS
rmsprop = keras.optimizers.RMSprop(
    lr=learning_rate,
    rho=rho,
    epsilon=None,
    decay=decay
)

rnnmodel.compile(
    loss="mse",
    optimizer=rmsprop
)

# ⬢⬢⬢⬢⬢➤ FIT
rnnmodel.fit(
    xTrain,
    yTrain,
    nb_epoch=epochs,
    verbose=1,
    shuffle=True,
    validation_split=validation_split
)

if save_model_flg:
    rnnmodel.save(os.path.join(save_model_path, '{}.h5'.format(model_name)))

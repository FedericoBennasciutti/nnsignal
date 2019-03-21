#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import os
import numpy as np
from scipy.io import loadmat, savemat
import matplotlib as mpl

from nnutils.rnn import RNNOfflineData, testRNN

mpl.use('TkAgg')  # or whatever other backend that you want
if mpl:
    import matplotlib.pyplot as plt

# ⬢⬢⬢⬢⬢➤ DATA

number_input_signals = 8
number_output_signals = 15
seq_len = 10

# extract
xtestdata_path = "/home/riccardo/tests/test_emg/data/xtest.mat"
xtestdata = loadmat(xtestdata_path)
xTest = xtestdata['test']

ytestdata_path = "/home/riccardo/tests/test_emg/data/ytest.mat"
ytestdata = loadmat(ytestdata_path)
yTest = ytestdata['test']

# prepare
rnndata = RNNOfflineData(seq_len, normalise=False, number_input_signals=number_input_signals)
xTest, yTest = rnndata.prepareData(xTest, yTest)


# ⬢⬢⬢⬢⬢➤ TEST


model = "/home/riccardo/MEGA/test_ml/emg/pippo.h5"

while True:
    chs = int(input("chs="))
    if chs == -1:
        break
    plotsetting = dict(signals=[chs])
    yPred_test = testRNN(model, xTest, yTest, "test_pippo", plotsetting=plotsetting)
    plt.show()

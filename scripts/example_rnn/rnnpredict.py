#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from nnutils.rnn import RNN

# ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇

''' WORK IN PROGRESS '''
# seq_len = 10
# model_h5_file_path = "/home/riccardo/MEGA/test_ml/emg/pippo.h5"
# rnn_reg = RNN(model_h5_file_path=model_h5_file_path,
#               bias_window_size=10,
#               seq_len=seq_len,
#               num_input_signals=8)

# bias = rnn_reg.getMean()
# data = [0]*10
# while True:
#     newdata = data[i]
#     rnn_reg.update(newdata)

#     if rnn_reg.isReady():
#         pred = rnn_reg.predict()-bias
#         print(pred)

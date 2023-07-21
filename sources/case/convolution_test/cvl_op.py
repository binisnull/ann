# -*- coding: UTF-8 -*-
"""
Author: BinIsNull

Contact: dengz004@163.com
"""
# Python 3.8.16

import numpy as np  # 1.18.5
import tensorflow as tf  # 2.3.0

import sources.main.convolution as cvl

# parameter
inp_sp = (14, 77, 89, 8)  # (number of inputs, width, height, depth)
fil_sp = (5, 5, 7, 8)  # (number of kernels, width, height, depth)
strid1 = 11
strid2 = 7
# inputs
inp = np.random.randint(-100, 100, inp_sp)
# filter
ken = np.random.randint(-50, 50, fil_sp)
# model
mod = cvl.ConvolutionalLayer((inp_sp[1], inp_sp[2], inp_sp[3]),
                             fil_sp[0], fil_sp[1], fil_sp[2], strid1, strid2)
mod.bias = np.zeros((1,))
mod.filter = ken
mod_res = mod.forward(inp)
# tensorflow
tfs_res = tf.nn.conv2d(inp,
                       np.concatenate((
                           mod.filter[0].reshape((fil_sp[1], fil_sp[2], fil_sp[3], 1)),
                           mod.filter[1].reshape((fil_sp[1], fil_sp[2], fil_sp[3], 1)),
                           mod.filter[2].reshape((fil_sp[1], fil_sp[2], fil_sp[3], 1)),
                           mod.filter[3].reshape((fil_sp[1], fil_sp[2], fil_sp[3], 1)),
                           mod.filter[4].reshape((fil_sp[1], fil_sp[2], fil_sp[3], 1))),
                           axis=3),
                       strides=[1, strid1, strid2, 1], padding='VALID')
# result
print(np.array_equal(mod_res, tfs_res))
print(tfs_res.shape)
print(mod_res.shape)

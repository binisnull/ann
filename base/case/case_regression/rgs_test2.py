# -*- coding: UTF-8 -*-
"""
Author: BinIsNull

Contact: dengz004@163.com
"""
# Python 3.8.16

import numpy as np

import base.tool.plot_tool as ptt
from base.main import activation_function as acf
from base.main.neural_network import MSERegressor


def func():
    """test function"""
    num = 400
    _x_ = np.random.uniform(-1, 1, num)
    _x_.sort()
    _y1_ = (_x_ ** 2) ** (1 / 3) + np.sqrt(1 - _x_ ** 2)
    _y2_ = (_x_ ** 2) ** (1 / 3) - np.sqrt(1 - _x_ ** 2)
    _y_ = np.array([_y1_, _y2_])
    return _x_.reshape((num, 1)), _y_.T


# data
train = func()
test = func()
# model
regressor = MSERegressor((1, 100, 50, 2),
                         (acf.leaky_relu, acf.tan_h,),
                         (acf.d_leaky_relu, acf.d_tan_h,))
# train
regressor.train_mul_epoch(train[0], train[1], 10, 100)
regressor.set_learning_rate(0.001)
regressor.train_mul_epoch(train[0], train[1], 10, 100)
regressor.set_learning_rate(0.0005)
regressor.train_mul_epoch(train[0], train[1], 10, 200)
regressor.set_learning_rate(0.0001)
regressor.train_mul_epoch(train[0], train[1], 10, 400)
ptt.plot_list_curve(regressor.loss, 'loss')
# test
train_x = train[0].reshape((train[0].shape[0],))
train_x = np.append(train_x, train_x)
train_y = train[1].T
train_y = train_y.reshape((train_y.shape[1] << 1, ))
predict_y = regressor.predict(train[0])
predict_y = predict_y.T
predict_y = predict_y.reshape((predict_y.shape[1] << 1, ))
test_x = test[0].reshape((test[0].shape[0],))
test_x = np.append(test_x, test_x)
test_y = regressor.predict(test[0])
test_y = test_y.T
test_y = test_y.reshape((test_y.shape[1] << 1, ))
ptt.plot_xy_curve(train_x, train_y, predict_y, test_x, test_y)

train_lab = np.ones(shape=(train_x.shape[0],)) * 0
predict_lab = np.ones(shape=(train_x.shape[0],))
test_lab = np.ones(shape=(test_x.shape[0],)) * 2
label = np.concatenate((train_lab, predict_lab, test_lab), axis=0)
x = np.concatenate((train_x, train_x, test_x), axis=0)
y = np.concatenate((train_y, predict_y, test_y), axis=0)
ptt.plot_distribution(x, y, label)

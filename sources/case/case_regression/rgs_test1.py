# -*- coding: UTF-8 -*-
"""
Author: BinIsNull

Contact: dengz004@163.com
"""
# Python 3.8.16

import numpy as np

import sources.tool.plot_tool as ptt
from sources.main import activation_function as acf
from sources.main.neural_network import MSERegressor


def func():
    """test function"""
    num = 400
    _x_ = np.random.uniform(-2, 2, num)
    _x_.sort()
    _y_ = np.e ** (np.sin(_x_)) / (_x_ + 2.1)
    return _x_.reshape((num, 1)), _y_.reshape((num, 1))


# data
train = func()
test = func()
# model
regressor = MSERegressor((1, 100, 50, 1),
                         (acf.leaky_relu, acf.tan_h,),
                         (acf.d_leaky_relu, acf.d_tan_h,))
# train
regressor.train_mul_epoch(train[0], train[1], 10, 1000)
regressor.set_learning_rate(0.001)
regressor.train_mul_epoch(train[0], train[1], 10, 1000)
regressor.set_learning_rate(0.0001)
regressor.train_mul_epoch(train[0], train[1], 10, 1000)
ptt.plot_list_curve(regressor.loss, 'loss')
# test
train_x = train[0].reshape((train[0].shape[0],))
train_y = train[1].reshape((train[1].shape[0],))
predict_y = regressor.predict(train[0])
predict_y = predict_y.reshape((predict_y.shape[0],))
test_x = test[0].reshape((test[0].shape[0],))
test_y = regressor.predict(test[0])
test_y = test_y.reshape((test_y.shape[0],))
ptt.plot_xy_curve(train_x, train_y, predict_y, test_x, test_y)
#
train_lab = np.ones(shape=(train_x.shape[0],)) * 0
predict_lab = np.ones(shape=(train_x.shape[0],))
test_lab = np.ones(shape=(test_x.shape[0],)) * 2
label = np.concatenate((train_lab, predict_lab, test_lab), axis=0)
x = np.concatenate((train_x, train_x, test_x), axis=0)
y = np.concatenate((train_y, predict_y, test_y), axis=0)
ptt.plot_distribution(x, y, label)

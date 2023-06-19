# -*- coding: UTF-8 -*-
"""
Author: BinIsNull

Contact: dengz004@163.com
"""
# Python 3.8.16

import random as rd

import numpy as np

import sources.main.activation_function as acf
import sources.main.neural_network as nn
import sources.tool.plot_tool as ptt


def generate_data(num):
    """Generate train and test data."""
    features = []
    labels = []
    for i in range(num):
        x = rd.uniform(-2, 2)
        y = rd.uniform(-2, 2)
        features.append([x, y])
        if x ** 2 + y ** 2 < 1:
            labels.append([1, 0, 0, 0, 0])
        else:
            if x < 0 and y < 0:
                labels.append([0, 0, 0, 0, 1])
            elif x > 0 > y:
                labels.append([0, 0, 0, 1, 0])
            elif x > 0 and y > 0:
                labels.append([0, 0, 1, 0, 0])
            else:
                labels.append([0, 1, 0, 0, 0])
    return np.array(features), np.array(labels)


# data
train_x, train_y = generate_data(60000)
test_num = 10000
test_x, test_y = generate_data(test_num)
# model
classifier = nn.SoftmaxClassifier((2, 50, 50, 5),
                                  (acf.leaky_relu, acf.tan_h,),
                                  (acf.d_leaky_relu, acf.d_tan_h,))
# train
classifier.set_learning_rate(0.1)
classifier.train_mul_epoch(train_x, train_y, 10, 2)
classifier.set_learning_rate(0.01)
classifier.train_mul_epoch(train_x, train_y, 10, 4)
classifier.set_learning_rate(0.001)
classifier.train_mul_epoch(train_x, train_y, 10, 8)
# test
print(classifier.test_accuracy(test_x, test_y) / test_num)
# plot
predict_y = classifier.predict(test_x)
ptt.plot_loss_acc(classifier.loss, classifier.accuracy)
ptt.plot_distribution(test_x[::, 0], test_x[::, 1], predict_y, cmap='rainbow')

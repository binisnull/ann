# -*- coding: UTF-8 -*-
"""
Author: BinIsNull

Contact: dengz004@163.com
"""
# Python 3.8.16

import pickle

import numpy as np

import sources.main.activation_function as acf
import sources.main.neural_network as nn
import sources.tool.plot_tool as ptt

# data
train_x = pickle.load(open('features', 'rb'))
train_y = pickle.load(open('labels', 'rb'))
# model
classifier = nn.SoftmaxClassifier((2, 41, 53, 3),
                                  (acf.leaky_relu, acf.sigmoid),
                                  (acf.d_leaky_relu, acf.d_sigmoid))
# train
classifier.set_learning_rate(0.08)
classifier.train_mul_epoch(train_x, train_y, 10, 5)
classifier.set_learning_rate(0.06)
classifier.train_mul_epoch(train_x, train_y, 10, 10)
classifier.set_learning_rate(0.01)
classifier.train_mul_epoch(train_x, train_y, 10, 10)
# plot
test_x = np.random.uniform(-1, 1, 100000).reshape((50000, 2))
predict_y = classifier.predict(test_x)
ptt.plot_loss_acc(classifier.loss, classifier.accuracy)
ptt.plot_distribution(test_x[::, 0], test_x[::, 1], predict_y)
pickle.dump(classifier, open('ift_model', 'wb'))

# -*- coding: UTF-8 -*-
"""
Author: BinIsNull

Contact: dengz004@163.com
"""
# Python 3.8.16

import pickle

import numpy as np

import sources.tool.plot_tool as ptt

# data
x_test = np.random.uniform(-1, 1, 100000).reshape((50000, 2))
# model
classifier = pickle.load(open('ift_model', 'rb'))
# plot
plt_label = classifier.predict(x_test)
ptt.plot_distribution(x_test[::, 0], x_test[::, 1], plt_label)

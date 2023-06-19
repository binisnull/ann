# -*- coding: UTF-8 -*-
"""
Author: BinIsNull

Contact: dengz004@163.com
"""
# Python 3.8.16

import numpy as np  # 1.18.5


def cross_entropy_loss(outputs, labels):
    """Cross-entropy loss function"""
    return -np.sum(
        labels * np.log(np.where(outputs == 0, 0.001, outputs))
    ) / labels.shape[0]


def d_cross_entropy_loss(outputs, labels):
    """The partial derivative of the cross-entropy loss function"""
    return -labels / np.where(
        outputs == 0, 0.001, outputs
    ) / labels.shape[0]


def mean_squared_loss(outputs, labels):
    """Mean squared loss function"""
    delta = outputs - labels
    return np.sum(delta * delta) / (outputs.shape[0] * outputs.shape[1])


def d_mean_squared_loss(outputs, labels):
    """The partial derivative of the mean squared loss function"""
    return 2 * (outputs - labels) / (outputs.shape[0] * outputs.shape[1])

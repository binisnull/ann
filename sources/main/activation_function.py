# -*- coding: UTF-8 -*-
"""
Author: BinIsNull

Contact: dengz004@163.com
"""
# Python 3.8.16

import numpy as np  # 1.18.5


def sigmoid(np2d_arr):
    """Sigmoid function"""
    return 1 / (1 + np.exp(-np2d_arr))


def d_sigmoid(np2d_arr):
    """The partial derivative of the sigmoid function"""
    sgm = sigmoid(np2d_arr)
    return sgm * (1 - sgm)


def relu(np2d_arr):
    """Rectified linear unit function"""
    return np.maximum(np2d_arr, 0)


def d_relu(np2d_arr):
    """The partial derivative of the rectified linear unit function"""
    return np.where(np2d_arr > 0, 1, 0)


def leaky_relu(np2d_arr):
    """Leaky rectified linear unit function"""
    return np.minimum(np2d_arr, 0) * 0.01 + np.maximum(np2d_arr, 0)


def d_leaky_relu(np2d_arr):
    """The partial derivative of the leaky rectified linear unit function"""
    mtx = np.where(np2d_arr < 0, np2d_arr, 1)
    return np.where(mtx > 0, mtx, 0.01)


def tan_h(np2d_arr):
    """Hyperbolic tangent function"""
    exp = np.exp(-2 * np2d_arr)
    return (1 - exp) / (1 + exp)


def d_tan_h(np2d_arr):
    """The partial derivative of the hyperbolic tangent function"""
    th = tan_h(np2d_arr)
    return 1 - th * th

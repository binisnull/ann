# -*- coding: UTF-8 -*-
"""
Author: BinIsNull

Contact: dengz004@163.com
"""
import numpy
# Python 3.8.16

import numpy as np  # 1.18.5


class ConvolutionalLayer(object):
    """Convolutional layer"""

    def __init__(self, feature_shape, kern_shape, stride=1):
        """
        Parameters
        --
        feature_shape: tuple or list
            (width, height, depth)
        kern_shape: tuple or list
            (num_kern, width, height)
        stride: integer
        """
        # Shape of convolutional kernel
        self.__k_shape = [i for i in kern_shape]
        self.__k_shape.append(feature_shape[-1])
        # Convolutional kernel
        self.__kern = np.random.uniform(
            -1, 1, self.__k_shape,
        )
        # Bias array
        self.__bias = np.random.uniform(-1, -1, (self.__k_shape[0],))
        # Output tensor width
        self.__otp_wdt = (feature_shape[0] - self.__k_shape[1]) // stride + 1
        # Output tensor height
        self.__otp_hgt = (feature_shape[1] - self.__k_shape[2]) // stride + 1
        # Row and column number pairs for local features
        num = self.__otp_wdt * self.__otp_hgt
        self.__inp_arr = np.arange(num)
        row = self.__inp_arr // self.__otp_wdt
        col = self.__inp_arr % self.__otp_wdt
        self.__inp_arr = ((np.append(row * stride, col * stride)).reshape((2, num))).T

    def __cal_element(self, arr, feature):
        row = arr[0]
        col = arr[1]
        elem = feature[row:row + self.__k_shape[2]:1, col:col + self.__k_shape[1]:1] * self.__kern
        return elem.sum(axis=3).sum(axis=2).sum(axis=1)

    def __cal_one_sample(self, feature):
        otp = np.apply_along_axis(self.__cal_element, 1, self.__inp_arr, feature)
        return otp.reshape((self.__otp_wdt, self.__otp_hgt, self.__k_shape[0])) + self.__bias

    def forward(self, features):
        """
        Parameters
        --
        features: numpy.ndarray
            shape=(num_sample, width, height, depth)
        """
        return np.apply_along_axis(
            lambda x, y: self.__cal_one_sample(y[x[0]]),
            1,
            np.arange(features.shape[0]).reshape((features.shape[0], 1)),
            features
        )


if __name__ == '__main__':
    pass

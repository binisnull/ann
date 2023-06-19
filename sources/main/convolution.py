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
            (width_feature, height_feature, depth)
        kern_shape: tuple or list
            (num_kernel, width_kernel, height_kernel)
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
        # The pair of row and column numbers of
        # the local feature tensors in the global feature tensor
        num = self.__otp_wdt * self.__otp_hgt
        self.__inp_arr = np.arange(num)
        row = self.__inp_arr // self.__otp_wdt
        col = self.__inp_arr % self.__otp_wdt
        self.__inp_arr = ((np.append(row * stride, col * stride)).reshape((2, num))).T

    def __cal_element(self, arr, feature):
        row_stt, col_stt = arr[0], arr[1]
        row_end, col_end = row_stt + self.__k_shape[2], col_stt + self.__k_shape[1]
        # (width, height, depth) * (num_kernel, width, height, depth)
        elem = feature[row_stt:row_end:1, col_stt:col_end:1] * self.__kern
        # (num_kernel, )
        return elem.sum(axis=3).sum(axis=2).sum(axis=1)

    def __cal_one_sample(self, feature):
        otp = np.apply_along_axis(self.__cal_element, 1, self.__inp_arr, feature)
        return otp.reshape((self.__otp_wdt, self.__otp_hgt, self.__k_shape[0])) + self.__bias

    def forward(self, features):
        """
        Parameters
        --
        features: numpy.ndarray
            shape=(num_sample, width_feature, height_feature, depth)
        """
        return np.apply_along_axis(
            lambda x, y: self.__cal_one_sample(y[x[0]]),
            1,
            np.arange(features.shape[0]).reshape((features.shape[0], 1)),
            features
        )


if __name__ == '__main__':
    pass

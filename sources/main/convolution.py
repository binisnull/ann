# -*- coding: UTF-8 -*-
"""
Author: BinIsNull

Contact: dengz004@163.com
"""
# Python 3.8.16

import numpy as np  # 1.18.5


class ConvolutionalLayer(object):
    """Convolutional layer"""

    def __init__(self, input_shape, num_kern, h_kern, w_kern, stride=1):
        """
        Parameters
        --
        input_shape: tuple or list
            (height, width, depth)
        num_kern: integer
            Number of kernel
        h_kern: integer
            Height of kernel
        w_kern: integer
            Width of kernel
        stride: integer
        """
        # Shape of convolutional kernel
        self.__k_shape = [num_kern, h_kern, w_kern, input_shape[-1]]
        # Convolutional kernel
        self.kernel = np.random.uniform(
            -1, 1, self.__k_shape,
        )
        # Bias array
        self.bias = np.random.uniform(-1, -1, (self.__k_shape[0],))
        # Output tensor width
        self.__w_out = (input_shape[1] - self.__k_shape[2]) // stride + 1
        # Output tensor height
        self.__h_out = (input_shape[0] - self.__k_shape[1]) // stride + 1
        # The pair of row and column numbers of
        # the local input tensors in the global input tensor
        num = self.__w_out * self.__h_out
        self.__locations = np.arange(num)
        row = self.__locations // self.__w_out
        col = self.__locations % self.__w_out
        self.__locations = ((np.append(row, col)).reshape((2, num))).T * stride
        # Parameters for next ConvolutionalLayer
        self.next_input_shape = (self.__h_out, self.__w_out, self.__k_shape[0])

    def __cal_element(self, location, np3d):
        row_stt, col_stt = location[0], location[1]
        row_end, col_end = row_stt + self.__k_shape[1], col_stt + self.__k_shape[2]
        elem = np3d[row_stt:row_end:1, col_stt:col_end:1] * self.kernel
        return elem.sum(axis=3).sum(axis=2).sum(axis=1)

    def __cal_one_sample(self, np3d):
        otp = np.apply_along_axis(self.__cal_element, 1, self.__locations, np3d)
        return otp.reshape((self.__h_out, self.__w_out, self.__k_shape[0])) + self.bias

    def forward(self, np4d):
        """
        Parameters
        --
        np4d: numpy.ndarray
            shape=(number, height, width, depth)
        """
        return np.apply_along_axis(
            lambda index, inputs: self.__cal_one_sample(inputs[index[0]]),
            1,
            np.arange(np4d.shape[0]).reshape((np4d.shape[0], 1)),
            np4d
        )


if __name__ == '__main__':
    pass

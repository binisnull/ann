# -*- coding: UTF-8 -*-
"""
Author: BinIsNull

Contact: dengz004@163.com
"""
# Python 3.8.16

import numpy as np  # 1.18.5


class ConvolutionalLayer(object):
    """Convolutional layer"""

    def __init__(self, input_shape, num_kern, h_kern, w_kern,
                 vtc_stride=1, hrz_stride=1):
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
        vtc_stride: integer
        hrz_stride: integer
        """
        # Shape of convolutional kernel
        self.__ksp = [num_kern, h_kern, w_kern, input_shape[-1]]
        # Convolutional kernel
        self.filter = np.random.uniform(
            -1, 1, self.__ksp,
        )
        # Bias array
        self.bias = np.random.uniform(-1, -1, (self.__ksp[0],))
        # Output tensor width
        self.__w_out = (input_shape[1] - self.__ksp[2]) // hrz_stride + 1
        # Output tensor height
        self.__h_out = (input_shape[0] - self.__ksp[1]) // vtc_stride + 1
        # The pair of row and column numbers of
        # the local input tensors in the global input tensor
        num = self.__w_out * self.__h_out
        idx_out = np.arange(num)
        row = (idx_out // self.__w_out) * vtc_stride
        col = (idx_out % self.__w_out) * hrz_stride
        self.__locations = ((np.append(row, col)).reshape((2, num))).T
        # Parameters for next ConvolutionalLayer
        self.next_input_shape = (self.__h_out, self.__w_out, self.__ksp[0])

    def __cal_elem(self, location, np3d):
        row_stt, col_stt = location[0], location[1]
        row_end, col_end = row_stt + self.__ksp[1], col_stt + self.__ksp[2]
        elem = np3d[row_stt:row_end:1, col_stt:col_end:1] * self.filter
        return elem.sum(axis=3).sum(axis=2).sum(axis=1)

    def __cal_one(self, np3d):
        rst = np.apply_along_axis(self.__cal_elem, 1, self.__locations, np3d)
        return rst.reshape((self.__h_out, self.__w_out, self.__ksp[0])) + self.bias

    def forward(self, np4d):
        """
        Parameters
        --
        np4d: numpy.ndarray
            shape=(number, height, width, depth)
        """
        return np.apply_along_axis(
            lambda index, inputs: self.__cal_one(inputs[index[0]]),
            1,
            np.arange(np4d.shape[0]).reshape((np4d.shape[0], 1)),
            np4d
        )


if __name__ == '__main__':
    pass

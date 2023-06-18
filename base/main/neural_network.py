# -*- coding: UTF-8 -*-
"""
Author: BinIsNull

Contact: dengz004@163.com
"""
# Python 3.8.16

import numpy as np  # 1.18.5


class FullyConnectedNN(object):
    """Fully connected neural network"""

    @property
    def num_layers(self):
        """Number of layers"""
        return self.__num_lyr

    def __init__(self, layer_dims, activ_funcs):
        # Number of layers
        self.__num_lyr = len(layer_dims)
        # Input matrices of hidden and output layers
        self.lyr_in = dict()
        # Output matrices of all layers
        self.lyr_out = {0: np.array([])}
        # Weight matrices of hidden and output layers
        self.weight = dict()
        # Bias column vectors of hidden and output layers
        self.bias = dict()
        # Activation functions of hidden and output layers
        self.__ac_func = dict()
        # Initialization
        for i in range(1, self.__num_lyr, 1):
            self.lyr_in[i] = np.array([])
            self.lyr_out[i] = np.array([])
            self.weight[i] = np.random.uniform(
                -1, 1, (layer_dims[i], layer_dims[i - 1])
            )
            self.bias[i] = np.random.uniform(-1, 1, (layer_dims[i], 1))
            self.__ac_func[i] = activ_funcs[i - 1]
        # Error matrix of the backward propagation process
        self.bp_err = np.array([])
        # Weight matrix stored for the backward propagation process before updating
        self.bp_wgt = np.array([])

    def grad_loss_out(self):
        """The partial derivatives of the loss function
        with respect to elements of an output matrix"""
        return self.bp_err @ self.bp_wgt

    def grad_loss_weight(self, layer_id):
        """The partial derivatives of the loss function
        with respect to elements of a weight matrix"""
        return self.bp_err.T @ self.lyr_out[layer_id - 1]

    def grad_loss_bias(self):
        """The partial derivatives of the loss function
        with respect to elements of a bias column vector"""
        err0_sum = self.bp_err.sum(0)
        return err0_sum.reshape((err0_sum.shape[0], 1))

    def forward(self, features):
        """Forward propagation"""
        self.lyr_out[0] = features
        for i in range(1, self.__num_lyr, 1):
            self.lyr_in[i] = (
                    self.weight[i] @ self.lyr_out[i - 1].T + self.bias[i]
            ).T
            self.lyr_out[i] = self.__ac_func[i](self.lyr_in[i])


class BackpropagationNN(FullyConnectedNN):
    """Back propagation (backward propagation of errors) neural network"""

    def set_learning_rate(self, rate):
        """Initialize the learning rate."""
        self.__lr = rate if 0 < rate < 1 else 0.01

    def cal_out_err(self):
        """Calculate the error matrix of the output layer."""
        pass

    def __init__(self, layer_dims, activ_funcs, hid_grad_activ_funcs):
        super().__init__(layer_dims, activ_funcs)
        # Label matrix
        self.label = None
        # Learning rate
        self.__lr = 0.01
        # Derivatives of all hidden layer activation functions
        self.__hid_grd_acf = dict()
        # Output layer id, a key of lyr_out
        self.max_lid = self.num_layers - 1
        for i in range(1, self.max_lid, 1):
            self.__hid_grd_acf[i] = hid_grad_activ_funcs[i - 1]

    def __adjust_weight(self, layer_id):
        self.weight[layer_id] -= self.__lr * self.grad_loss_weight(layer_id)

    def __adjust_bias(self, layer_id):
        self.bias[layer_id] -= self.__lr * self.grad_loss_bias()

    def __cal_hid_err(self, layer_id):
        """Calculate an error matrix of a hidden layer."""
        self.bp_err = self.__hid_grd_acf[layer_id](self.lyr_in[layer_id]) * self.grad_loss_out()
        self.bp_wgt = self.weight[layer_id]

    def __backward(self):
        self.bp_err = self.cal_out_err()
        self.bp_wgt = self.weight[self.max_lid]
        self.__adjust_weight(self.max_lid)
        self.__adjust_bias(self.max_lid)
        for i in range(self.max_lid - 1, 0, -1):
            self.__cal_hid_err(i)
            self.__adjust_weight(i)
            self.__adjust_bias(i)

    def train_one_batch(self, features, labels):
        """Train one batch."""
        self.forward(features)
        self.label = labels
        self.__backward()

    def train_one_epoch(self, features, labels, batch_size):
        """Train one epoch."""
        num_sp = labels.shape[0]
        iteration = (num_sp + batch_size - 1) // batch_size
        index = np.arange(0, num_sp, 1)
        # Disrupt the entire training set order.
        # And this is actually not mandatory to execute.
        np.random.shuffle(index)
        ite_idx = 0
        for i in range(iteration):
            sub_idx = index[ite_idx:ite_idx + batch_size:1]
            sub_features = features[sub_idx]
            sub_labels = labels[sub_idx]
            self.train_one_batch(sub_features, sub_labels)
            ite_idx += batch_size


class SoftmaxClassifier(BackpropagationNN):
    """Classifier based on cross-entropy & softmax"""

    @staticmethod
    def softmax(np2d_arr):
        """Softmax function"""
        row_max = np.amax(np2d_arr, axis=1)
        exp_matrix = np.exp(np2d_arr - row_max.reshape((row_max.shape[0], 1)))
        row_sum = np.sum(exp_matrix, axis=1)
        return exp_matrix / row_sum.reshape((row_sum.shape[0], 1))

    @staticmethod
    def ce_loss(outputs, labels):
        """Cross-entropy loss function"""
        return -np.sum(labels * np.log(np.where(outputs == 0, 0.001, outputs))) / labels.shape[0]

    def cal_out_err(self):
        """Calculate the error matrix of the output layer."""
        return (self.lyr_out[self.max_lid] - self.label) / self.label.shape[0]

    def __init__(self, layer_dims, hid_activ_funcs, hid_grad_activ_funcs):
        """
        Parameters
        --
        layer_dims: tuple or list
            Numbers of neurons in each layer
        hid_activ_funcs: tuple or list
            Activation functions of all hidden layers
        hid_grad_activ_funcs: tuple or list
            Derivatives of all hidden layer activation functions
        """
        activ_funcs = [func for func in hid_activ_funcs]
        activ_funcs.append(self.softmax)
        super().__init__(layer_dims, activ_funcs, hid_grad_activ_funcs)
        self.loss = []
        self.accuracy = []

    def train_mul_epoch(self, features, labels, batch_size, epoch_size,
                        test_features=None, test_labels=None):
        """Train multiple epochs."""
        test_x = features if test_features is None else test_features
        test_y = labels if test_labels is None else test_labels
        for i in range(epoch_size):
            self.train_one_epoch(features, labels, batch_size)
            #
            # ** **
            _features = test_x
            _labels = test_y
            self.forward(_features)
            loss = self.ce_loss(self.lyr_out[self.max_lid], _labels)
            outputs = np.argmax(self.lyr_out[self.max_lid], axis=1)
            _labels = np.argmax(_labels, axis=1)
            accu = np.sum(outputs == _labels) / _labels.shape[0]
            print('{: <6}'.format(i), end='')
            print('{: <15.6f}'.format(loss), end='')
            print('{:.6f}'.format(accu))
            self.loss.append(loss)
            self.accuracy.append(accu)
            # ** **
            #

    def predict(self, features):
        """
        Returns
        --
        k_id: numpy.ndarray
            The ordinal numbers that distinguish their corresponding sample labels
        """
        self.forward(features)
        return np.argmax(self.lyr_out[self.max_lid], axis=1)

    def test_accuracy(self, features, labels):
        """
        Returns
        --
        num: integer
            The total number of samples whose prediction results matched their labels
        """
        self.forward(features)
        outputs = np.argmax(self.lyr_out[self.max_lid], axis=1)
        labels = np.argmax(labels, axis=1)
        return np.sum(outputs == labels)


class MSERegressor(BackpropagationNN):
    """Regressor based on mean squared error"""

    @staticmethod
    def ms_loss(outputs, labels):
        """Mean squared loss function"""
        delta = outputs - labels
        return np.sum(delta * delta) / (outputs.shape[0] * outputs.shape[1])

    def cal_out_err(self):
        """Calculate the error matrix of the output layer."""
        outs = self.lyr_out[self.max_lid]
        return 2 * (outs - self.label) / (outs.shape[0] * outs.shape[1])

    def __init__(self, layer_dims, hid_activ_funcs, hid_grad_activ_funcs):
        """
        Parameters
        --
        layer_dims: tuple or list
            Numbers of neurons in each layer
        hid_activ_funcs: tuple or list
            Activation functions of all hidden layers
        hid_grad_activ_funcs: tuple or list
            Derivatives of all hidden layer activation functions
        """
        activ_funcs = [func for func in hid_activ_funcs]
        activ_funcs.append(lambda arr: arr)
        super().__init__(layer_dims, activ_funcs, hid_grad_activ_funcs)
        self.loss = []

    def train_mul_epoch(self, features, labels, batch_size, epoch_size,
                        test_features=None, test_labels=None):
        """Train multiple epochs."""
        test_x = features if test_features is None else test_features
        test_y = labels if test_labels is None else test_labels
        for i in range(epoch_size):
            self.train_one_epoch(features, labels, batch_size)
            #
            # ** **
            _features = test_x
            _labels = test_y
            self.forward(_features)
            loss = self.ms_loss(self.lyr_out[self.max_lid], _labels)
            print('{: <6}'.format(i), end='')
            print('{: <15.6f}'.format(loss))
            self.loss.append(loss)
            # ** **
            #

    def predict(self, features):
        """
        Returns
        --
        k_id: numpy.ndarray
        """
        self.forward(features)
        return self.lyr_out[self.max_lid]

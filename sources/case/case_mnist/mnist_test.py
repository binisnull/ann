# -*- coding: UTF-8 -*-
"""
Author: BinIsNull

Contact: dengz004@163.com
"""
# Python 3.8.16

import numpy as np  # 1.18.5
from tensorflow.keras.datasets import mnist  # tensorflow-gpu 2.3.0
import sources.main.activation_function as af
import sources.main.neural_network as nn


def get_one_hot(one_dim_digit):
    """Represent a digit with a One-hot."""
    one_hot = np.zeros(shape=(10,))
    one_hot[one_dim_digit[0]] = 1
    return one_hot


# DATA

(train_features, train_labels), (test_features, test_labels) = mnist.load_data()
feature_dim = train_features.shape[1] * train_features.shape[2]
# train data
train_num = train_features.shape[0]
train_features = train_features.reshape((train_num, feature_dim)) / 255
train_labels = np.apply_along_axis(
    get_one_hot, axis=1, arr=train_labels.reshape((train_num, 1))
)
# test data
test_num = test_features.shape[0]
test_features = test_features.reshape((test_num, feature_dim)) / 255
test_labels = np.apply_along_axis(
    get_one_hot, axis=1, arr=test_labels.reshape((test_num, 1))
)

# MODEL

# There are 81 neurons in the hidden layer.
classifier = nn.SoftmaxClassifier((feature_dim, 81, 10),
                                  (af.sigmoid,),
                                  (af.d_sigmoid,))

# TRAINING

classifier.set_learning_rate(0.55)
# batch_size = 9, epoch_size = 11
classifier.train_mul_epoch(train_features, train_labels, 9, 11,
                           test_features, test_labels)
classifier.set_learning_rate(0.1)
# batch_size = 27, epoch_size = 9
classifier.train_mul_epoch(train_features, train_labels, 27, 9,
                           test_features, test_labels)
classifier.set_learning_rate(0.001)
# batch_size = 31, epoch_size = 13
classifier.train_mul_epoch(train_features, train_labels, 31, 13,
                           test_features, test_labels)

# TEST

print('\naccuracy = ', end='')
print(classifier.test_accuracy(test_features, test_labels) / test_num)

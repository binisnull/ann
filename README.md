# A multi-layer perceptron framework based only on numpy

## :feet:<font color=#783CB4>Introduction:umbrella:
This is a basic multi-layer perceptron framework based on numpy only, and you can flexibly modify the number of layers, the number of neurons, and the activation function of each layer according to your needs.
## :crystal_ball:<font color=#52367B>Usage:watch:
```
# -*- coding: UTF-8 -*-
"""
File Name: test_mnist.py
"""
# Python 3.8.16
from tensorflow.keras.datasets import mnist  # tensorflow-gpu 2.3.0
import numpy as np  # 1.18.5
import neural_network as nn
import activation_function as af


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

# Create a two-layer network where
# the hidden layer has 51 neurons,
# the output layer has 10 neurons,
# and the hidden layer activation function is the Sigmoid function
classifier = nn.SoftmaxClassifier((feature_dim, 51, 10),
                                  (af.sigmoid,),
                                  (af.d_sigmoid,))

# TRAINING

classifier.set_learning_rate(0.05)
classifier.train_mul_epoch(train_features, train_labels, 10, 11)
classifier.set_learning_rate(0.01)
classifier.train_mul_epoch(train_features, train_labels, 10, 13)

# TEST

print('\naccuracy = ', end='')
print(classifier.test_accuracy(test_features, test_labels) / test_num)

```

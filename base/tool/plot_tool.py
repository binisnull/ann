# -*- coding: UTF-8 -*-
"""
Author: BinIsNull

Contact: dengz004@163.com
"""
# Python 3.8.16

import matplotlib.pyplot as plt


def plot_list_curve(arr, name):
    """Plot curve."""
    plt.figure(figsize=(6, 5))
    plt.xlabel('epoch')
    plt.ylabel(name)
    plt.plot(arr, c='red', label=name)
    plt.legend()
    plt.show()


def plot_loss_acc(loss, accuracy):
    """Plot loss and accuracy curves."""
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(loss, c='red', label='cross-entropy loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.plot(accuracy, c='green', label='accuracy')
    plt.legend()
    plt.show()


def plot_distribution(xs, ys, labels, size=10, cmap='viridis'):
    """Plot a scatter plot."""
    plt.figure(figsize=(8, 5))
    plt.scatter(xs, ys, c=labels,
                marker='.', cmap=cmap, s=size)
    plt.colorbar()
    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()


def plot_xy_curve(x_train, y_train, y_predict, x_test, y_test):
    """Plot the regression curve."""
    plt.figure(figsize=(8, 5))
    plt.plot(x_test, y_test, label='test value', c='red')
    plt.plot(x_train, y_train, label='actual value', c='green')
    plt.plot(x_train, y_predict, label='predicted value', c='yellow')
    plt.show()

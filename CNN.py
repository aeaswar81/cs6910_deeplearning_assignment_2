# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 17:19:09 2021

@author: Arun-PC
"""

import tensorflow as tf

from tensorflow.keras import datasets, layers, models


class CNN:
    def __init__(self, input_shape, filters, kernel_size, strides=None, conv_activation=None, pool_size=None,
                 dense_layer_size=None, dense_activation=None, output_layer_size=None, output_activation=None):
        self.input_shape = input_shape
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.conv_activation = conv_activation
        self.pool_size = pool_size
        self.dense_layer_size = dense_layer_size
        self.dense_activation = dense_activation
        self.output_layer_size = output_layer_size
        self.output_activation = output_activation
        self.model = None
        self.history = None
        self.initCNN()

    def initCNN(self):
        self.model = models.Sequential()
        if self.strides is None:
            self.strides = [(1, 1)] * len(self.filters)
        # add convolution layers
        for i in range(len(self.filters)):
            self.model.add(layers.Conv2D(self.filters[i], self.kernel_size[i], strides=self.strides[i],
                                         activation=self.conv_activation, input_shape=self.input_shape))
            self.model.add(layers.MaxPooling2D(self.pool_size[i]))

        # flatten and add dense layers
        self.model.add(layers.Flatten())
        for i in range(len(self.dense_layer_size)):
            self.model.add(layers.Dense(self.dense_layer_size[i], activation=self.dense_activation[i]))

        # add output layer
        self.model.add(layers.Dense(self.output_layer_size, activation=self.output_activation))

    def fit(self, optimizer, loss, metrics, trainX, trainY, epochs, validation_data):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self.history = self.model.fit(trainX, trainY, epochs=epochs, validation_data=validation_data)
        return self.history

    def evaluate(self, testX, testY):
        return self.model.evaluate(testX,  testY, verbose=2)


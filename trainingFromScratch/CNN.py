# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 17:19:09 2021

@author: Arun-PC
"""

import tensorflow as tf

from tensorflow.keras import datasets, layers, models, Model
import matplotlib.pyplot as plt
import random
import tensorflow.keras.backend as K
import numpy as np


class CNNClassifier:
    def __init__(self, input_shape, filters, kernel_size, strides=None, conv_activation=None, pool_size=None,
                 dense_layer_size=None, dense_activation=None, output_layer_size=None, output_activation=None,
                 batch_normalization=False, dropout=None):
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
        self.batch_normalization = batch_normalization
        self.dropout = dropout
        self.model = None
        self.history = None
        self.initCNN()

    def initCNN(self):
        self.model = models.Sequential()
        if self.strides is None:
            self.strides = [(1, 1)] * len(self.filters)

        # add the first convolution layer with input size
        self.model.add(layers.Conv2D(self.filters[0], self.kernel_size[0], strides=self.strides[0],
                                     activation=None, input_shape=self.input_shape, name="conv_1"))
        # add batch norm layer before activation as suggested in the original paper
        if self.batch_normalization:
            self.model.add(layers.BatchNormalization())
        self.model.add(layers.Activation(self.conv_activation))
        self.model.add(layers.MaxPooling2D(self.pool_size[0]))

        # add the remaining convolution layers
        for i in range(len(self.filters) - 1):
            self.model.add(layers.Conv2D(self.filters[i], self.kernel_size[i], strides=self.strides[i],
                                         activation=None, name="conv_%d"%(i+2)))
            # add batch norm layer before activation as suggested in the original paper
            if self.batch_normalization:
                self.model.add(layers.BatchNormalization())
            self.model.add(layers.Activation(self.conv_activation))
            self.model.add(layers.MaxPooling2D(self.pool_size[i]))

        # flatten and add dense layers
        self.model.add(layers.Flatten())
        for i in range(len(self.dense_layer_size)):
            self.model.add(layers.Dense(self.dense_layer_size[i], activation=None))
            # add batch norm layer before activation as suggested in the original paper
            if self.batch_normalization:
                self.model.add(layers.BatchNormalization())
            self.model.add(layers.Activation(self.dense_activation))
            # add dropouts at the fully connected layers
            if self.dropout is not None:
                self.model.add(layers.Dropout(self.dropout))

        # add output layer
        self.model.add(layers.Dense(self.output_layer_size, activation=None))
        # add batch norm layer before activation as suggested in the original paper
        if self.batch_normalization:
            self.model.add(layers.BatchNormalization())
        self.model.add(layers.Activation(self.output_activation))

    def fit(self, optimizer, loss, metrics, trainX, trainY, epochs, validation_data):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self.history = self.model.fit(trainX, trainY, epochs=epochs, validation_data=validation_data)
        return self.history
    
    def fit_generator(self, optimizer, loss, metrics, train_generator, epochs, validation_generator, callbacks=None):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self.history = self.model.fit(train_generator, epochs=epochs, validation_data=validation_generator, callbacks=callbacks)
        return self.history
    
    def evaluate(self, testX, testY):
        return self.model.evaluate(testX,  testY, verbose=2)
    
    def evaluate_generator(self, test_generator):
        return self.model.evaluate(test_generator, verbose=2)
    
    def summary(self):
        print('printing summary of the model')
        self.model.summary()

    def deprocess_image(self, x):
        """Same normalization as in:
          https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
          """
        # normalize tensor: center on 0., ensure std is 0.25
        x = x.copy()
        x -= x.mean()
        x /= (x.std() + K.epsilon())
        x *= 0.25

        # clip to [0, 1]
        x += 0.5
        x = np.clip(x, 0, 1)

        # convert to RGB array
        x *= 255
        if K.image_data_format() == 'channels_first':
            x = x.transpose((1, 2, 0))
        x = np.clip(x, 0, 255).astype('uint8')
        return x

    def guidedbackprop(self, img):
        @tf.custom_gradient
        # define the new gradient
        def guidedRelu(x):
            def grad(dy):
                return tf.cast(dy > 0, "float32") * tf.cast(x > 0, "float32") * dy

            return tf.nn.relu(x), grad

        # create a sub model upto the desired layer here it is cov_3 later
        gb_model = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer("conv_5").output]
        )
        # find all layers which has activation as attribute
        layer_dict = [layer for layer in gb_model.layers[1:] if hasattr(layer, 'activation')]
        # change the activation to guidedrelu
        for layer in layer_dict:
            if layer.activation == tf.keras.activations.relu:
                layer.activation = guidedRelu
        # gradient tape to record the values to find gradient later
        with tf.GradientTape(persistent=True) as tape:
            # taking the 0th image as input
            inputs = tf.cast(img.reshape([-1, 256, 256, 3]), tf.float32)
            # watch the values of input
            tape.watch(inputs)
            # forward prop it through the sub model
            outputs = gb_model(inputs)
            # to find out non zero entries
            zero = tf.constant(0, dtype=tf.float32)
            locations = tf.not_equal(outputs, zero)
            # find their indices
            indices = tf.where(locations)
            # create a tensor array for storing the layer with only the non zero neuron
            ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
            # sample 10 neurons from the non zero list of neuron indices
            ind_set = random.sample(range(0, len(indices)), 10)
            # loop through the indices set
            for i in ind_set:
                # create a zero matrix
                z = np.zeros_like(outputs)
                # set the entry with index value same as that of the non zero neuron
                z[tuple(indices[i].numpy())] = 1
                # multiply this with the z matrix , we are doing this since we cant directly modify tensors
                # hence using elementwise multiplication with zero matrix as a work around
                # write it to the array
                ta = ta.write(i, tf.math.multiply(outputs, z))
        # define a tensor array for gradients of each of these neurons
        tg = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        # for each of the neurons find the gradient
        for i in ind_set:
            tg = tg.write(i, tape.gradient(ta.read(i), inputs)[0])
        # plotting
        # plt.imshow((np.array(grads)))
        fig = plt.figure(figsize=(15, 15))
        columns = 1
        rows = 10
        # fig.subplots_adjust(hspace=0.2)
        for i in range(1, columns * rows + 1):
            fig.add_subplot(rows, columns, i)
            plt.imshow(np.flip(self.deprocess_image(np.array(tg.read(ind_set[i - 1]))), -1))
        plt.tight_layout()
        plt.show()


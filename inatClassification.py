# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 10:14:01 2021

@author: Arun-PC
"""

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from CNN import CNNClassifier

data_augment_args = dict(rotation_range=90,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)

datagen = ImageDataGenerator(validation_split=0.1, **data_augment_args)

train_iterator = datagen.flow_from_directory('inaturalist_12K/train/',
                                             subset="training", target_size=(512, 512))
val_iterator = datagen.flow_from_directory('inaturalist_12K/train/',
                                           subset="validation", target_size=(512, 512))

datagen = ImageDataGenerator()
test_iterator = datagen.flow_from_directory('inaturalist_12K/val/', target_size=(512, 512))

batchX, batchy = train_iterator.next()
print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))

input_shape = batchX.shape[1:4]
filters = [64, 64, 64]
kernel_size = [(3, 3), (3, 3), (3, 3)]
conv_activation = 'relu'
pool_size = [(2, 2), (2, 2), (2, 2)]
dense_layer_size = [64]
dense_activation = ['relu']
output_layer_size = 10
output_activation = 'softmax'

cnn = CNNClassifier(input_shape, filters, kernel_size, conv_activation=conv_activation, pool_size=pool_size,
                    dense_layer_size=dense_layer_size, dense_activation=dense_activation,
                    output_layer_size=output_layer_size, output_activation=output_activation)
history = cnn.fit_generator(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'], train_generator=train_iterator,
                  epochs=10, validation_generator=val_iterator)

test_loss, test_acc = cnn.evaluate_generator(test_iterator)
print(test_acc)
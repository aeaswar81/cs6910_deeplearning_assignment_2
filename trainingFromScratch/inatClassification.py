# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 10:14:01 2021

@author: Arun-PC
"""

import preprocess_crop  # to add support for cropping of images
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from CNN import CNNClassifier

data_augment_args = dict(rotation_range=90,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.2)
target_size = (256, 256)

datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.1, **data_augment_args)

train_iterator = datagen.flow_from_directory('inaturalist_12K/train/',
                                             subset="training", target_size=target_size,
                                             interpolation='lanczos:center')
val_iterator = datagen.flow_from_directory('inaturalist_12K/train/',
                                           subset="validation", target_size=target_size,
                                           interpolation='lanczos:center')

datagen = ImageDataGenerator(rescale=1. / 255)
test_iterator = datagen.flow_from_directory('inaturalist_12K/val/', target_size=target_size,
                                            interpolation='lanczos:center')

batchX, batchy = train_iterator.next()
print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))

input_shape = batchX.shape[1:4]
filters = [64]*6
kernel_size = [(3, 3)] * 6
conv_activation = 'relu'
pool_size = [(2, 2)]*6
dense_layer_size = [64]
dense_activation = 'relu'
output_layer_size = 10
output_activation = 'softmax'
batch_norm = True
dropout = 0.2
EPOCHS = 5

cnn = CNNClassifier(input_shape, filters, kernel_size, conv_activation=conv_activation, pool_size=pool_size,
                    dense_layer_size=dense_layer_size, dense_activation=dense_activation,
                    output_layer_size=output_layer_size, output_activation=output_activation,
                            batch_normalization=batch_norm, dropout=dropout)

cnn.guidedbackprop(batchX[0])
# history = cnn.fit_generator(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
#                             metrics=['accuracy'], train_generator=train_iterator,
#                             epochs=EPOCHS, validation_generator=val_iterator)

# test_loss, test_acc = cnn.evaluate_generator(test_iterator)
# print(test_acc)

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 21:30:43 2021

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
batch_size = 64

datagen = ImageDataGenerator(validation_split=0.1, **data_augment_args)

train_iterator = datagen.flow_from_directory('inaturalist_12K/train/', batch_size=batch_size,
                                             subset="training", target_size=target_size,
                                             interpolation='lanczos:center')
val_iterator = datagen.flow_from_directory('inaturalist_12K/train/',
                                           subset="validation", batch_size=batch_size,
                                           target_size=target_size,
                                           interpolation='lanczos:center')

datagen = ImageDataGenerator()
test_iterator = datagen.flow_from_directory('inaturalist_12K/val/', batch_size=batch_size,
                                            target_size=target_size,
                                            interpolation='lanczos:center')

batchX, batchy = train_iterator.next()
print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))

input_shape = batchX.shape[1:4]
filters = [64,64,128,128,256,256]
kernel_size = [(3, 3)] * len(filters)
conv_activation = 'relu'
pool_size = [(2, 2)] * len(filters)
dense_layer_size = [128] * 2
dense_activation = ['relu']
output_layer_size = 10
output_activation = 'softmax'
batch_norm = True
dropout = 0.3
EPOCHS = 25
earlyStopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)

cnn = CNNClassifier(input_shape, filters, kernel_size, conv_activation=conv_activation, pool_size=pool_size,
                    dense_layer_size=dense_layer_size, dense_activation=dense_activation,
                    output_layer_size=output_layer_size, output_activation=output_activation,
                            batch_normalization=batch_norm, dropout=dropout)
history = cnn.fit_generator(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                            metrics=['accuracy'], train_generator=train_iterator,
                            epochs=EPOCHS, validation_generator=val_iterator,
                            callbacks=[earlyStopping])

#save the model
print('saving model...')
cnn.model.save('cnn_best_model')

test_loss, test_acc = cnn.evaluate_generator(test_iterator)
print('Test accuracy', test_acc)

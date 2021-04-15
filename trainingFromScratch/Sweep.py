# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 19:32:40 2021

@author: Arun-PC
"""

import wandb
from wandb.keras import WandbCallback
import preprocess_crop  # to add support for cropping of images
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from CNN import CNNClassifier

fo = {'same': 1, 'double': 2, 'halve': 0.5}


#!wandb login

def sweep():
    run = wandb.init()
    config = wandb.config
    name = str(config.no_of_filters) + "_" + str(config.filter_organization) + "_" \
           + ('conv=%d' % config.no_of_conv_layers) + "_" \
           + ('dense=%d*%d' % (config.dense_layer_size, config.no_of_dense_layers)) + "_" \
           + ('batch=%d' % config.batch_size) + "_" \
           + ('image=%d' % config.image_size) + "_" \
           + ('dropout=%.2f' % config.dropout if config.dropout is not None else '') + "_" \
           + ('epochs=%d' % config.epochs) \
           + ('_augment' if config.data_augmentation == 'yes' else '') \
           + ('_batchnorm' if config.batch_normalization == 'yes' else '')

    wandb.run.name = name
    data_augment_args = dict(rotation_range=90,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2)
    target_size = (config.image_size,) * 2

    datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.1, **data_augment_args) if config.data_augmentation == 'yes' \
        else ImageDataGenerator(validation_split=0.1)

    train_iterator = datagen.flow_from_directory('/content/drive/MyDrive/inaturalist_12K/train/', batch_size=config.batch_size,
                                                 subset="training", target_size=target_size,
                                                 interpolation='lanczos:center')
    val_iterator = datagen.flow_from_directory('/content/drive/MyDrive/inaturalist_12K/train/', batch_size=config.batch_size,
                                               subset="validation", target_size=target_size,
                                               interpolation='lanczos:center')

    datagen = ImageDataGenerator(rescale=1. / 255)
    test_iterator = datagen.flow_from_directory('/content/drive/MyDrive/inaturalist_12K/val/', batch_size=config.batch_size,
                                                target_size=target_size,
                                                interpolation='lanczos:center')

    batchX, batchy = train_iterator.next()
    print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))

    input_shape = batchX.shape[1:4]
    print('Input shape : ', input_shape)
    filters = [config.no_of_filters] * config.no_of_conv_layers
    multiplier = 1
    for i, v in enumerate(filters):
        filters[i] = max(int(v * multiplier), 1) # keep atleast 1 filter in a layer
        multiplier *= fo[config.filter_organization]
    print('Filters : ', filters)
    kernel_size = [(config.kernel_size,)*2] * config.no_of_conv_layers
    print('Kernel : ', kernel_size)
    conv_activation = 'relu' # use the standard relu activation for convolution layers
    pool_size = [(2, 2)] * config.no_of_conv_layers  # use the standard pool size for all layers
    dense_layer_size = [config.dense_layer_size]*config.no_of_dense_layers
    print('Dense layers : ', dense_layer_size)
    dense_activation = 'relu' # always keep relu activation for dense layers
    output_layer_size = 10  # no of classes
    output_activation = 'softmax' # softmax for classification problem
    batch_norm = True if config.batch_normalization == 'yes' else False
    dropout = config.dropout
    EPOCHS = config.epochs

    cnn = CNNClassifier(input_shape, filters, kernel_size, conv_activation=conv_activation, pool_size=pool_size,
                        dense_layer_size=dense_layer_size, dense_activation=dense_activation,
                        output_layer_size=output_layer_size, output_activation=output_activation,
                        batch_normalization=batch_norm, dropout=dropout)
    print('CNN Network created. Starting training...')
    history = cnn.fit_generator(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                                metrics=['accuracy'], train_generator=train_iterator,
                                epochs=EPOCHS, validation_generator=val_iterator,
                                callbacks=[WandbCallback()])

    test_loss, test_acc = cnn.evaluate_generator(test_iterator)
    wandb.log({'test_accuracy': test_acc, 'test_loss': test_loss})
    run.finish()


sweep_config = {
    'method': 'bayes',  # grid, random
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'epochs': {
            'values': [5, 10]
        },
        'batch_size': {
            'values': [32, 64, 128]
        },
        'image_size': {
            'values': [256, 512]
        },
        'no_of_conv_layers': {
            'values': [5, 6]
        },
        'no_of_filters': {
            'values': [16, 32, 64, 128]
        },
        'kernel_size': {
            'values': [3, 4]
        },
        'dense_layer_size': {
            'values': [64, 128]
        },
        'no_of_dense_layers': {
            'values': [1, 2, 3]
        },
        'filter_organization': {
            'values': ['same', 'double', 'halve']
        },
        'data_augmentation': {
            'values': ['yes', 'no']
        },
        'dropout': {
            'values': [None, 0.15, 0.2, 0.25, 0.3]
        },
        'batch_normalization': {
            'values': ['yes', 'no']
        }
    }
}

# wandb.init(project='cs6910-assignment1', name = 'class-samples-1')
sweep_id = wandb.sweep(sweep_config, project="cs6910-assignment2-test")

# %% start wandb sweep
wandb.agent(sweep_id, sweep, count=1)

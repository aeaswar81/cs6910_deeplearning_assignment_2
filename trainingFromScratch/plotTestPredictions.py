# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 22:43:24 2021

@author: Arun-PC
"""

import tensorflow as tf
from tensorflow import keras
import preprocess_crop  # to add support for cropping of images
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from collections import defaultdict
import numpy as np


def plotGrid(samples, n_row, n_col):    
    fig, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=(15,30))
    
    # for row in axes:
    #     for col in row:
    #         col.axis('off')
    
    # for ax, col in zip(axes[0], samples.keys()):
    #     ax.set_title(col)
        
    for ax, row in zip(axes[:,0], samples.keys()):
        ax.set_ylabel(row, labelpad=30, rotation=0, size='x-large')
        ax.yaxis.label.set_color('blue')

    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=0.1,wspace=0.1)
    i = 0
    r=0
    for k, v in samples.items():
        c=0
        for image, pred_label in v:
            #plt.subplot(n_row, n_col, i + 1)
            # plt.imshow(image)
            #plt.xticks(())
            #plt.yticks(())
            axes[r][c].imshow(image)
            axes[r][c].set_title(pred_label, color='red' if k!=pred_label else 'green')
            axes[r][c].set_xticks([])
            axes[r][c].set_yticks([])
            c += 1
            i += 1
        r += 1

    fig.tight_layout()
    plt.show()


datagen = ImageDataGenerator()
test_iterator = datagen.flow_from_directory('inaturalist_12K/val/', batch_size=128,
                                            target_size=(256, 256),
                                            interpolation='lanczos:center')

batchX, batchy = test_iterator.next()
labelMapping = {v: k for k, v in test_iterator.class_indices.items()}
labels = np.argmax(batchy, axis=1)
labels = [labelMapping[i] for i in labels]
count = defaultdict(lambda: 0)
samples = {}

# load the model
model = keras.models.load_model('cnn_best_model')

for x, y in zip(batchX, labels):
    if count[y] == 0:
        samples[y] = []
    if count[y] < 3:
        p = model.predict(np.expand_dims(x, axis=0))
        pred_label = labelMapping[np.argmax(p)]
        samples[y].append((x.astype(np.uint8), pred_label))
        count[y] += 1

    # if gathered 3 images for all classes break
    flag = True
    for v in count.values():
        if v != 3:
            flag = False

    if flag:
        break



plotGrid(samples, 10, 3)

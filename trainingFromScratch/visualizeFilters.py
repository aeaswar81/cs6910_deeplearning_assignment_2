# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 21:23:33 2021

@author: Arun-PC
"""
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import random

def plotGrid(images, n_row, n_col):
	plt.figure()
	plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.1, wspace=.1)
	for i in range(n_row * n_col):
		plt.subplot(n_row, n_col, i + 1)
		plt.imshow(images[:,:,i], cmap=plt.cm.gray)
		plt.xticks(())
		plt.yticks(())
	plt.show()

# load the model
model = keras.models.load_model('cnn_best_model')
# print('Model summary')
# model.summary()
filters, biases = model.layers[0].get_weights()

# visualize the filters
# add the filters along depth dimension
filters = np.sum(filters, axis = 0)

# normalize the values between 0 to 1
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

# plot the filters in a grid
plotGrid(filters, 8, 8)

first_layer = tf.keras.Model(inputs=model.inputs, outputs=model.layers[0].output)
first_layer.summary()
#%%
import random
datagen = ImageDataGenerator()
test_iterator = datagen.flow_from_directory('inaturalist_12K/val/', batch_size=128,
                                            target_size=(256, 256),
                                            interpolation='lanczos:center')

batchX, batchy = test_iterator.next()
sample = batchX[random.randint(0, len(batchX))]
sample = np.expand_dims(sample, axis=0)
out = first_layer.predict(sample)


# plot all 64 maps in an 8x8 squares
square = 8
ix = 1
for _ in range(square):
	for _ in range(square):
		# specify subplot and turn of axis
		ax = plt.subplot(square, square, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		# plot filter channel in grayscale
		plt.imshow(out[0, :, :, ix-1], cmap='gray')
		ix += 1
# show the figure
plt.show()

plt.axis('off')
plt.imshow(sample[0].astype(np.uint8), interpolation='nearest')
plt.show()

#%%
model.summary()
model.guidedbackprop(batchX[0])

import tensorflow as tf
from tensorflow.keras import datasets
from CNN import CNNClassifier
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

input_shape = (32, 32, 3)
no_of_conv_layers = 3
filters = [128] * no_of_conv_layers
kernel_size = [(3, 3)] * no_of_conv_layers
conv_activation = 'relu'
pool_size = [(2, 2)] * no_of_conv_layers
dense_layer_size = [64]
dense_activation = 'relu'
output_layer_size = 10
output_activation = 'softmax'
validation_data = (test_images, test_labels)
EPOCHS = 5

cnn = CNNClassifier(input_shape, filters, kernel_size, conv_activation=conv_activation, pool_size=pool_size,
                    dense_layer_size=dense_layer_size, dense_activation=dense_activation,
                    output_layer_size=output_layer_size, output_activation=output_activation)
cnn.summary()

history = cnn.fit(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'], trainX=train_images, trainY=train_labels,
                  epochs=EPOCHS, validation_data=validation_data)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = cnn.evaluate(test_images, test_labels)
print(test_acc)

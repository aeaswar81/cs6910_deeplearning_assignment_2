This project uses Object oriented programming for modularity. There are three parts namely
  1. Training CNN from Scratch(trainingFromScratch)
  2. Transfer Learning(transferLearning)
  3. Application using YoloV3(yolov3App)

# Part A - Training CNN from Scratch
For the purpose of part A, we have created a class **CNNClassifier** using tensorflow and keras. Its constructor can take the following parameters:
  1. input_shape - Dimension of the input
  2. filters - List of number of filters in every layer
  3. kernel_size - List of size of kernel in every layer
  4. strides - Stride size to be used in every layer. By default stride size is (1,1)
  5. conv_activation - Activation function to be used in convolution layers
  6. pool_size - Pooling size. By default, pool size is (2,2)
  7. dense_layer_size - List of size of dense layers
  8. dense_activation - Activation function to be used at the dense layers
  9. output_layer_size - Size of the output layer. This should be set equal to the number of classes.
  10. output_activation - Activation function to be used at the output layer. By default, it is set as 'softmax'
  11. batch_normalization - Whether batch normalization has to be used or not. Boolean value. If true, batch normalization is done at every layer.
  12. dropout - The value of dropout to be used at the dense layers.

This class has the following methods:
## fit(optimizer, loss, metrics, trainX, trainY, epochs, validation_data)
To train the CNNClassifier.

optimizer - optimizer to be used while training
loss - loss function to be used for training
metrics - metrics to be monitored while training
trainX - training data
trainY - training labels
epochs - number of epochs until which the model will be trained
validation_data - validation data to be used

## fit_generator(optimizer, loss, metrics, train_generator, epochs, validation_generator, callbacks)

This is same as fit except that it uses keras data generator as input instead of arrays. It also has a callback parameter.

## evaluate(testX, testY)
To evaluate the model on test data

## evaluate_generator(test_generator)
To evaluate the model using a test data generator

## summary()
To print the summary of the model

## guidedbackprop(image)
This method generates plot that shows which part of the image excites various neurons in the fifth convolution layer.

This part also the following python scripts:
  1. runCNN.py - loads cifar10 dataset, builds a simple CNNClassifier, trains the model, plots accuracy vs epoch, and evaluates the model on test data
  2. inatClassification.py - loads the inaturalist dataset, builds a CNNClassifier, trains the model, plot the guided backpropogation, and evaluates the model on test data
  3. Sweep.py - Runs wandb sweeps for hyperparameter tuning
  4. bestCNNModel.py - Trains and evaluates the best CNN configuration obtained from wandb sweeps
  5. preprocess_crop.py - To add support for center cropping of images as a preprocessing step
  6. visualizeFilters.py - To visualize the filters and feature maps of a layer
  7. plotTestPredictions.py - To plot samples of test data and the predictions made by a model

# Part B - Transfer Learning
The python notebook has code for loading various pretrained models and fine tuning them. This also can run sweeps for the same.

# Part C - Application using YoloV3
This part has a separate readme file.

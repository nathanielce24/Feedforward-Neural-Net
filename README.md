# **Feedforward Dense Neural Network for Grayscale Image Recognition**

This is my first attempt at building a neural network from the ground up. In it's current state it's not particularly optimized or efficient, but with a very low learning rate and enough epochs you can achieve very accurate results. Currently a new model must be trained every time, but I intend to implent save/load functionality in the near future.

How to Train a Model:
* Download src file
* Modify epochs and learning rate in train.py
* Run train.py, and see how accurate the model is.

## Technical Overview
DenseLayer:
* Represents the connections between two layers
* Initializes weights and biases
* Handels forwardpropogation and backpropogation for a single layer

NeuralNet:
* Creates an array of DenseLayer objects given network architecture
* Handels forward propogation and backpropogation for the entire network
* Trains and evaluates the model

Train:
* Downloads and preprocesses mnist dataset
* Creates and trains network given architecture, epochs, and learning rate.


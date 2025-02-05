# **Feedforward Dense Neural Network for Handwritten Digit Recognition**

This is my first attempt at building a neural network from the ground up. In it's current state it's not particularly optimized or efficient, but with a low enough learning rate and enough epochs you can achieve very accurate results. Currently a new model must be trained every time, but I intend to implent Save/Load model functionality in the future.

How to Train a Model:
* Download src file
* Modify epochs and learning rate in train.py
* Run train.py, and see how accurate your model is.

## Technical Overview
DenseLayer:
* Represents the connections between two layers
* Initializes weights and biases
* Handels forwardpropogation and backpropogation for a single layer

NeuralNet:
* Creates an array of DenseLayer objects given network architecture
* Handels forward propogation and backpropogation for the entire network
* Trains and evaluates the model


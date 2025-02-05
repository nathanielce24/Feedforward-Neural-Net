# **Feedforward Dense Neural Network for Grayscale Image Classification**

This is my first attempt at building a neural network from the ground up. In it's current state it's not particularly optimized or efficient, but with a very low learning rate and enough epochs you can achieve very accurate results. 

How to Train a Model:
* Download src
* Modify epochs and learning rate in one of the provided training files
* Run training file, and see how accurate the model is.

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

## Training Files:
TrainFashion: Trains a model to differentiate between 10 different articles of clothing

# **Feedforward Dense Neural Network for Grayscale Image Classification**

This is my first attempt at building a neural network from the ground up. In it's current state it's not very optimized or efficient, but with a very low learning rate and enough iterations you can achieve accurate results. 

How to Train a Model:
* Download src
* Pip install medmnist package for datasets
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
* **TrainPneumonia:** Trains a model on the PneumoniaMNIST dataset to identify Pneumonia in pediactric X-Rays (≈ 80% Accuracy).
* **Train Organ:** Trains a model on  the OrganCMNIST dataset to identify 11 organs (≈70% Accuracy).
* **TrainBreastCancer:** Trains a model on the BreastMNIST dataset to identify breast cancer in ultrasounds (≈80% Accuracy).

![alt text](https://github.com/nathanielce24/Feedforward-Neural-Net/blob/main/Flowcharts/DenseLayer-Class.png?raw=true)
![alt text](https://github.com/nathanielce24/Feedforward-Neural-Net/blob/main/Flowcharts/NeuralNet-Class.png?raw=true)
![alt text](https://github.com/nathanielce24/Feedforward-Neural-Net/blob/main/Flowcharts/Training-Process.png?raw=true)

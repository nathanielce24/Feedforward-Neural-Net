# **Feedforward Dense Neural Network for Grayscale (Biomedical) Image Classification**

This is a feedforwad, fully connected neural network that I built from the ground up. By tweaking the learning rate and iterations, I have been able to achieve above average results for biomedical image classification from MNIST datasets. Currently a new model has to be trained every time, but I plan to implement save/load functionality in the near future.
How to Train a Model:
* Download src
* Pip install medmnist package for datasets
* Modify 'epochs' (technically iterations) and learning rate in one of the provided training files
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
* **TrainPneumonia:** Trains a model on the PneumoniaMNIST dataset to identify Pneumonia in pediactric X-Rays (≈ 86% Accuracy).
* **Train Organ:** Trains a model on  the OrganCMNIST dataset to identify 11 organs (≈70% Accuracy).
* **TrainBreastCancer:** Trains a model on the BreastMNIST dataset to identify breast cancer in ultrasounds (≈82% Accuracy).

Note that while these results are certaintly better than guessing (and above average for this architecture), class imbalancees in the data may cause a large amount of false negatives that aren't captured in the accuracy (for pneumonia and breast-cancer.)

## 
![alt text](https://github.com/nathanielce24/Feedforward-Neural-Net/blob/main/Flowcharts/DenseLayer-Class.png?raw=true)
![alt text](https://github.com/nathanielce24/Feedforward-Neural-Net/blob/main/Flowcharts/NeuralNet-Class.png?raw=true)
![alt text](https://github.com/nathanielce24/Feedforward-Neural-Net/blob/main/Flowcharts/Training-Process.png?raw=true)

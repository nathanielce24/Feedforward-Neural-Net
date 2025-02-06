# **Feedforward Dense Neural Network for Grayscale (Biomedical) Image Classification**

This is a feedforwad, fully connected neural network that I built from the ground up. By tweaking the learning rate and iterations, I have been able to achieve accurate results for various biomedical image classifications. 

While a convolutional neural network would undoubtedly perform better, I wanted to start from scratch so I opted for a very basic architecture. Given the limitations of DNN's, I'm extremely happy with the results.

## How to Train a Model:
* Download src
* Pip install medmnist package for datasets
* Modify 'epochs' (technically iterations) and learning rate in one of the provided training files
* Run the training file, and watch loss decrease over the iterations.

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
* **TrainPneumonia:** Trains a model on the PneumoniaMNIST dataset to identify Pneumonia in pediactric X-Rays (≈86% Accuracy).
* **TrainOrgan:** Trains a model on  the OrganCMNIST dataset to classify 11 different organs in abdominal CT scans (≈70% Accuracy).
* **TrainBreastCancer:** Trains a model on the BreastMNIST dataset to identify breast cancer in ultrasounds (≈82% Accuracy).



Note that, while these results are better than random (and above average for this architecture), the models success at binary classification is likely exagerated due to class imbalences in the data that cause the network to overvalue false negatives.
##
##
![alt text](https://github.com/nathanielce24/Feedforward-Neural-Net/blob/main/Flowcharts/PneumImg.png?raw=true)
## 
##
![alt text](https://github.com/nathanielce24/Feedforward-Neural-Net/blob/main/Flowcharts/DenseLayer.png?raw=true)
![alt text](https://github.com/nathanielce24/Feedforward-Neural-Net/blob/main/Flowcharts/NeuralNet-Class.png?raw=true)
![alt text](https://github.com/nathanielce24/Feedforward-Neural-Net/blob/main/Flowcharts/Training-Process.png?raw=true)


import random
import numpy as np
import math 
import numpy as np

class DenseLayer:
    def __init__(self, input_size, layer_size, activation_function):
        self.input_size = input_size
        self.layer_size = layer_size
        self.activation_function = activation_function
        self.weights = self.initialize_weights()
        self.biases = np.zeros((1, layer_size))

    def initialize_weights(self):
       # return np.random.randn(self.input_size, self.layer_size) * 0.01  # Small random values
        return np.random.randn(self.input_size, self.layer_size) * np.sqrt(2/self.layer_size) #HEinitialization

    def change_weights(self, gradient):     #Subtracts a provided gradient from weights
        self.weights -= gradient

    def change_biases(self, gradient):      #Subtracts a provided gradientfrom biases
        self.biases -= gradient 

    def calculate_weighted_sums(self, inputs):
        return np.dot(inputs, self.weights) + self.biases

    def forward(self, inputs):
        return self.activation_function(self.calculate_weighted_sums(inputs)) 

    def back_pass(self, weighted_sums, prev_activations, activations, expected_activation=None, prev_gradient=None, is_output=False):  
        if is_output:  
            weighted_sums_gradient = activations - expected_activation #weighted sum gradient for final layer, softmax activation instead of sigmoid
        else:
            weighted_sums_gradient = prev_gradient * self.sigmoid_derivative(weighted_sums) #

    
        weight_gradients = np.dot(prev_activations.T, weighted_sums_gradient)
        bias_gradients = np.sum(weighted_sums_gradient, axis=0, keepdims=True)

        return weight_gradients, bias_gradients, weighted_sums_gradient



    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
        

    @staticmethod
    def sigmoid_derivative(x):         #sigmoid derivative used to find how much activation changes with respect to weighted sum
        sigmoid= DenseLayer.sigmoid(x)
        return sigmoid * (1 - sigmoid)
        

    @staticmethod
    def softmax(x):        #takes weighted sums and converts to probabilities
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)


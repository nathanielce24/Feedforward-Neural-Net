import random
import numpy as np
import math 

class DenseLayer:
     def __init__ (self, input_size, layer_size, activation_function):
        self.input_size = input_size
        self.layer_size = layer_size
        self.activation_function = activation_function
        
        self.weights = self.initialize_weights()
        self.biases = np.zeros((1,layer_size))             #2d row vector or biases, initially 0
        
     
     def set_weights(self, weights):
        self.weights = weights

     def set_biases(self, biases):
        self.biases = biases

     def initialize_weights(self):
        return np.random.rand(self.input_size, self.layer_size)  * 0.01    #Columns are neurons, rown are weights, initialized randomly
     
     def calculate_weighted_sums(self, inputs):
        weighted_sums = np.dot(inputs, self.weights) + self.biases   #Dot products of input activations and weights, + biases
        #print(weighted_sums)
        return weighted_sums
     
     def forward(self, inputs):
         outputs = self.sigmoid(self.calculate_weighted_sums(inputs))     #apply activation function to get the inputs for the next layer
         #print(outputs)
         return outputs
        
     def back_pass(self, inputs, expected_activation):  
        activation = self.forward(inputs)
        print(activation)
        weighted_sums_gradient = (activation - expected_activation) * self.sigmoid_derivative(self.calculate_weighted_sums(inputs))  #how far weighted sums are from loss = 0
        #print(weighted_sums_gradient)
        weight_gradients = np.dot(inputs.T, weighted_sums_gradient)   #how far weights are from loss = 0
        bias_gradients = np.sum(weighted_sums_gradient, axis=0, keepdims=True)  #how far biases are from loss = 0
        return weight_gradients, bias_gradients
     
     def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
     def sigmoid_derivative(self, x):
        return self.sigmoid(x) - (1-self.sigmoid(x))
   
     
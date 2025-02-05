
from DenseLayer import DenseLayer
import random
import numpy as np
from DenseLayer import DenseLayer

class NeuralNet:
    def __init__(self, architecture, activation_function, loss_function, learning_rate): #Architecure is array of neuron counts ex. [128,64,10]
        self.architecture = architecture
        self.activation_function = activation_function
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.network_activations = []
        self.network = self.create_network(architecture)  

    def create_network(self, architecture):
        network = []
        for i in range(len(architecture)-1):
            activation = DenseLayer.sigmoid  # sigmoid activation function
            if i == len(architecture) - 2:  # Output layer (Dense layers are connections between neurons, so one less dense layers than layer of neurons)
                activation = DenseLayer.softmax  #converts final weighted sums to probabilities w/ softmax fucntion
            network.append(DenseLayer(architecture[i], architecture[i+1], activation))  #(inputs from previous layer, number of neurons in current layer, activation function)
        #network.append(DenseLayer(128, 10, DenseLayer.softmax))
        return network

    def forward_pass(self, input):
        self.network_activations = [input]   #Initial activations are the pixels of the input file
        for layer in self.network:
            input = layer.forward(input)     #feed the input forward to get activations going into next layer
            self.network_activations.append(input)  #update full network activations
        return input


    def back_propagate(self, expected_activations):
        activations = self.network_activations  
        weighted_sums = [layer.calculate_weighted_sums(activations[i]) for i, layer in enumerate(self.network)]  #calculating the weighted sums based on activation for each layer

        weight_gradients = []
        bias_gradients = []

        #handeling final output layer
        weight_gradient, bias_gradient, prev_gradient = self.network[-1].back_pass(weighted_sums[-1],  #Current weighted sums
                                                                                    activations[-2],   #Previous activations
                                                                                    activations[-1],   #Current Activations
                                                                                    expected_activation=expected_activations, 
                                                                                    is_output=True)    #To account for different activation in final layer
        weight_gradients.append(weight_gradient)
        bias_gradients.append(bias_gradient)

        
        #handeling hidden layers
        for i in range(len(self.network) - 2, -1, -1): #starts at layer before output layer, iterates backwards
            prev_gradient = np.dot(prev_gradient, self.network[i + 1].weights.T)  # Takes last error gradient and pushes backwards through weights to get current error gradient
                                                                                  #Transpose neccesary because we're moving backwards through the layers now
            weight_gradient, bias_gradient, prev_gradient = self.network[i].back_pass(weighted_sums[i], #Current layers weighted sums
                                                                                      activations[i],   #current lauers activations
                                                                                      activations[i + 1], #activations for last kayer
                                                                                      prev_gradient=prev_gradient, 
                                                                                      is_output=False)

            weight_gradients.insert(0, weight_gradient)  #Stored in beggining because we're working backwards
            bias_gradients.insert(0, bias_gradient)

    
        for i, layer in enumerate(self.network):
            layer.weights -= self.learning_rate * weight_gradients[i]   #gradient is how far away we are from expected w/b, so we need to subtract 
            layer.biases -= self.learning_rate * bias_gradients[i]




    def train(self, input, expected, epochs):
        for epoch in range(epochs):            
            predictions = self.forward_pass(input)   #forward pass to get current models predictions
            loss = self.cross_entropy_loss(predictions, expected)  #calculate loss
            print(f"EPOCH {epoch} - Loss: {loss:.4f}")
            self.back_propagate(expected)   #update model based on expected values

    def mean_squared_error_loss(self,predictions, expected):
      return np.mean((expected - predictions) ** 2)
    
    def cross_entropy_loss(self, predictions, expected):
        return -np.mean(np.sum(expected * np.log(predictions), axis=1)) + 0.0000000000000000000001   #add small value bc 0 causes problems 


    def evaluate(self, input, expected):
        predictions = self.forward_pass(input)  
        predicted_classes = np.argmax(predictions, axis=1) #Maximum probability class is models prediction
        actual_classes = np.argmax(expected, axis=1)  #Actual class is one hot encoded
        accuracy = np.mean(predicted_classes == actual_classes)  #How often actual class = predicted class
        return accuracy
    
    def save_model(self):
        pass

    def load_model(self):
        pass 


     
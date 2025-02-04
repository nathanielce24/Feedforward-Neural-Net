class DenseLayer:
     def __init__ (self, input_size, layer_size, activation_function):
        self.input_size = input_size
        self.layer_size = layer_size
        self.activation_function = activation_function
        
        self.weights = self.initialize_weights(self.input_size, self.layer_size) 
        self.biases = None 
        
     
     def set_weights(self, weights):
        self.weights = weights

     def set_biases(self, biases):
        self.biases = biases

     def initialize_weights(self, input_size, layer_size):
        return None
     
     def calculate_weighted_sum(self, inputs):
        weighted_sums = np.dot(inputs, self.weights) + self.biases
        return weighted_sums
     
     def forward(self, inputs):
        if self.activation_function: 
           outputs = self.activation_function(self.calculate_weighted_sum(inputs))
           return outputs
        
        else: 
           return self.calculate_weighted_sum(inputs)
        

     def back_pass(self, loss, learning_rate):
        pass
     
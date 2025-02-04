from DenseLayer import DenseLayer
import numpy as np

nn = DenseLayer(5,5, activation_function=DenseLayer.sigmoid)

input = nn.initialize_weights()



print(nn.back_pass(np.array([[0,0,0,0,0]]), np.array([[0,0,0,1,0]])))
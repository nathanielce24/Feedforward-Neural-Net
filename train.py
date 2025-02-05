
#Example of a training process that should yield a model with >95% accuracy

from DenseLayer import DenseLayer
from NeuralNet import NeuralNet
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0
X_train = X_train.reshape(X_train.shape[0], 784)
X_test = X_test.reshape(X_test.shape[0], 784)
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)


architecture = [784, 128, 64, 10] 
learning_rate = 0.00005  #tends to get stuck with anything higher 


nn = NeuralNet(architecture, DenseLayer.sigmoid, "cross_entropy", learning_rate)

nn.train(X_train, y_train, epochs=300)  #

accuracy = nn.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")
import numpy as np
from tensorflow.keras.utils import to_categorical
from NeuralNet import NeuralNet
from medmnist import OrganCMNIST
from DenseLayer import DenseLayer

train_dataset = OrganCMNIST(split="train", download=True) #Splits training and testing data
test_dataset = OrganCMNIST(split="test", download=True)

# Extract images and labels
X_train, y_train = train_dataset.imgs, train_dataset.labels  #Images and labels for images
X_test, y_test = test_dataset.imgs, test_dataset.labels

X_train = X_train.astype(np.float32) / 255.0 #converts from 0-255rgb value to 0-1 pixel activation
X_test = X_test.astype(np.float32) / 255.0
X_train = X_train.reshape(X_train.shape[0], 784)  #Flatterened from 2d matrix to 1d array
X_test = X_test.reshape(X_test.shape[0], 784)
y_train = to_categorical(y_train, num_classes=11) #one hot encodes labels
y_test = to_categorical(y_test, num_classes=11)


architecture = [784, 128, 64, 11] 
learning_rate = 0.0005  #tends to get stuck with anything higher 


nn = NeuralNet(architecture, DenseLayer.sigmoid, "cross_entropy", learning_rate)

nn.train(X_train, y_train, epochs=500)

accuracy = nn.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

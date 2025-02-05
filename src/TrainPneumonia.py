import numpy as np
from tensorflow.keras.utils import to_categorical
from NeuralNet import NeuralNet
from medmnist import PneumoniaMNIST
from DenseLayer import DenseLayer


train_dataset = PneumoniaMNIST(split="train", download=True)
test_dataset = PneumoniaMNIST(split="test", download=True)

# Extract images and labels
X_train, y_train = train_dataset.imgs, train_dataset.labels
X_test, y_test = test_dataset.imgs, test_dataset.labels

X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0
X_train = X_train.reshape(X_train.shape[0], 784)
X_test = X_test.reshape(X_test.shape[0], 784)
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)


architecture = [784, 128, 64, 2] 
learning_rate = 0.00005  #tends to get stuck with anything higher 


nn = NeuralNet(architecture, DenseLayer.sigmoid, "cross_entropy", learning_rate)

nn.train(X_train, y_train, epochs=1000)

accuracy = nn.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

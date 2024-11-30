import numpy as np
from data import get_mnist
from Activations import Activations
from tqdm import tqdm

class Layer:
    def __init__(self, shape, learning_rate=0.01, activation='sigmoid'):
        self.id = 'None'
        self.name = 'Base'
        self.weight = np.random.randn(shape[0], shape[1]) * np.sqrt(2.0 / shape[0])
        self.bias = np.zeros((1, shape[1]))
        self.learning_rate = learning_rate
       
        if (activation not in Activations):
            raise ValueError(f'Invalid activation function provided: {activation}')
        
        self.activation = Activations[activation]['main']
        self.activationDerivative = Activations[activation]['derivative']
        
        self.delta = None
        self.input = None
        self.z = None
        self.a = None
        
        
        
    def __str__(self):
        return f"Layer {self.id}\n"
    
    def forward(self, X):
        self.input = X
        self.z = np.dot(X, self.weight) + self.bias
        self.a = self.activation(self.z)
        return self.a
    
    def backward(self, *args):
        delta = self.delta
        self.weight -= self.learning_rate * np.dot(self.input.T, delta) / self.input.shape[0]
        self.bias -= self.learning_rate * np.sum(delta, axis=0, keepdims=True) / self.input.shape[0]
        return delta
    
    def calculateDelta(self, *args):
        pass

class Dense(Layer):
    def __init__(self, shape, learning_rate=0.01, activation='sigmoid'):
        super().__init__(shape, learning_rate, activation)
        self.name = 'NN_DENSE'
        
    def calculateDelta(self, succeedingLayer):
        self.delta = np.dot(succeedingLayer.delta, succeedingLayer.weight.T) * self.activationDerivative(self.z)

class Output(Layer):
    def __init__(self, shape, learning_rate=0.01, activation='sigmoid'):
        super().__init__(shape, learning_rate, activation)
        self.name = 'NN_OUTPUT'
        
    def calculateDelta(self, Y):
        self.delta = self.a - Y

def predict(X, y):
    forwardInput = X
    for layer in layers:
        forwardInput = layer.forward(forwardInput)
    predictions = forwardInput
    accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y, axis=1))
    print(f"Accuracy: {accuracy * 100:.2f}%")

# Load data
x_train_images, x_test_images, y_train_labels, y_test_labels = get_mnist()

# Network architecture
arch = [784, 128, 10]
layers = []

for i in range(len(arch) - 1):
    shape = (arch[i], arch[i + 1])
    
    if (i == len(arch) - 2):
        layers.append(Output(shape, learning_rate=0.1))
    else:
        layers.append(Dense(shape, learning_rate=0.1))
        
epochs = 10
batch_size = 32

for epoch in range(epochs):
    print(f"{epoch + 1}/{epochs} Epochs")
    shuffle_idx = np.random.permutation(len(x_train_images))
    X_shuffled = x_train_images[shuffle_idx]
    y_shuffled = y_train_labels[shuffle_idx]
    
    for i in tqdm(range(0, len(x_train_images), batch_size), desc="Batches", ncols=100, leave=True):
        batch_x = X_shuffled[i:i+batch_size]
        batch_y = y_shuffled[i:i+batch_size]
        
        forwardInput = batch_x
        for layer in layers:
            forwardInput = layer.forward(forwardInput)
        
        for i in range(len(layers) - 1, -1, -1):
            if (i == len(layers) - 1):
                layers[i].calculateDelta(batch_y)
            else:
                layers[i].calculateDelta(layers[i + 1])
            
            layers[i].backward()

    predict(x_train_images, y_train_labels)
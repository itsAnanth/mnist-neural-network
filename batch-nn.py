import numpy as np
from data import get_mnist
from Activations import Activations
from tqdm import tqdm

class Layer:
    def __init__(self, shape, learning_rate=0.01, activation='sigmoid', id=None):
        self.id = id
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
        return f"Layer_{self.name}_{self.id}"
    
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
    def __init__(self, shape, learning_rate=0.01, activation='sigmoid', id=None):
        super().__init__(shape, learning_rate, activation, id)
        self.name = 'NN_DENSE'
        
    def calculateDelta(self, succeedingLayer):
        self.delta = np.dot(succeedingLayer.delta, succeedingLayer.weight.T) * self.activationDerivative(self.z)

class Output(Layer):
    def __init__(self, shape, learning_rate=0.01, activation='sigmoid', id=None):
        super().__init__(shape, learning_rate, activation, id)
        self.name = 'NN_OUTPUT'
        
    def calculateDelta(self, Y):
        self.delta = self.a - Y
        
class NeuralNetwork:
    def __init__(self, layers=[], epochs=10, learning_rate=0.1, batch_size=32):
        self.epochs = epochs
        
        # applied to all layers when NN is created from a list of numbers representing neurons for each layer
        self.learning_rate = learning_rate
        self.layers = layers
        self.batch_size = batch_size
        
        
    def layers_from_list(self, layers):
        self.layers = layers
    
    def layers_from_narray(self, layersMap):
        for i in range(len(layersMap) - 1):
            shape = (layersMap[i], layersMap[i + 1])
            
            if (i == len(layersMap) - 2):
                self.layers.append(Output(shape, learning_rate=self.learning_rate, id=i))
            else:
                self.layers.append(Dense(shape, learning_rate=self.learning_rate, id=i))
                
        for layer in self.layers:
            print(layer)
            
    def predict(self, X, y):
        forwardInput = X
        for layer in self.layers:
            forwardInput = layer.forward(forwardInput)
        predictions = forwardInput
        accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y, axis=1))
        print(f"Accuracy: {accuracy * 100:.2f}%")
            
    def train(self, X, y):
        for epoch in range(self.epochs):
            print(f"{epoch + 1}/{self.epochs} Epochs")
            shuffle_idx = np.random.permutation(len(X))
            X_shuffled = X[shuffle_idx]
            y_shuffled = y[shuffle_idx]
            
            for i in tqdm(range(0, len(X), self.batch_size), desc="Batches", ncols=100, leave=True):
                batch_x = X_shuffled[i:i+self.batch_size]
                batch_y = y_shuffled[i:i+self.batch_size]
                
                forwardInput = batch_x
                for layer in self.layers:
                    forwardInput = layer.forward(forwardInput)
                
                for i in range(len(self.layers) - 1, -1, -1):
                    if (i == len(self.layers) - 1):
                        self.layers[i].calculateDelta(batch_y)
                    else:
                        self.layers[i].calculateDelta(self.layers[i + 1])
                    
                    self.layers[i].backward()

            self.predict(X, y)
        



# Load data
x_train_images, x_test_images, y_train_labels, y_test_labels = get_mnist()

# Network architecture
arch = [784, 128, 10]
nn = NeuralNetwork(
    epochs=10,
    learning_rate=0.1,
    batch_size=32
)
nn.layers_from_narray(arch)
nn.train(x_train_images, y_train_labels)

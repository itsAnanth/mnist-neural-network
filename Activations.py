import numpy as np
from data import get_mnist

def leakyRelu(x, alpha):
    return x if x > 0 else alpha * x

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
def sigmoid_derivative(z):
    a = sigmoid(z)
    return a * (1 - a)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

Activations = {
    'relu': {
        'main': relu,
        'derivative': relu_derivative
    },
    'sigmoid': {
        'main': sigmoid,
        'derivative': sigmoid_derivative
    },
    'softmax': {
        'main': softmax
    }
}

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from data import get_mnist
import pickle


# Load the MNIST data (ensure images are flattened and labels are one-hot encoded)
x_train_images, x_test_images, y_train_labels, y_test_labels = get_mnist()

epochs = 3
learning_rate = 0.01
w_i_h = np.random.randn(20, 784) * np.sqrt(2. / 784)
w_h_o = np.random.randn(10, 20) * np.sqrt(2. / 20)   # weights from hidden to output
b_i_h = np.zeros(shape=(20, 1))                  # biases for hidden layer
b_h_o = np.zeros(shape=(10, 1))                  # biases for output layer
correct_predictions = 0

def ReLu(x):
    return np.maximum(0, x)

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

# Sigmoid derivative function (corrected typo)
def sigmoid_derivative(x):
    return x * (1 - x)

# # Loop over epochs
# for epoch in range(epochs):
#     correct_predictions = 0
#     for i, (image, label) in enumerate(zip(x_train_images, y_train_labels)):
        
#         image = image.reshape(784, 1)  # Flatten image if necessary
#         label = label.reshape(10, 1)   # Ensure label is of shape (10, 1)
        
#         # Forward pass: input to hidden
#         hidden_pre_activation = (w_i_h @ image) + b_i_h
#         hidden_activated = sigmoid(hidden_pre_activation)
        
#         # Forward pass: hidden to output
#         output_pre_activation = (w_h_o @ hidden_activated) + b_h_o
#         output_activated = sigmoid(output_pre_activation)
        
#         # Prediction
#         # print(np.argmax(output_activated), np.argmax(label))
#         correct_predictions += int(np.argmax(output_activated) == np.argmax(label))
        
#         # Backpropagation
#         # Error at the output layer
#         output_delta = (output_activated - label) #* sigmoid_derivative(output_activated)
#         output_gradient = -learning_rate * output_delta @ hidden_activated.T
        
#         # Error at the hidden layer
#         hidden_delta = (w_h_o.T @ output_delta) * sigmoid_derivative(hidden_activated)
#         hidden_gradient = -learning_rate * hidden_delta @ image.T
        
#         # Update weights and biases
#         w_h_o += output_gradient
#         w_i_h += hidden_gradient
#         b_h_o += learning_rate * output_delta
#         b_i_h += learning_rate * hidden_delta
        
#         if (i % 10000 == 0):
#             print(f"Epoch {epoch + 1} | {i}/{len(x_train_images)} samples processed")
    
#     # Calculate accuracy after each epoch
#     print(f"Epoch {epoch+1}/{epochs} - Acc: {round((correct_predictions / x_train_images.shape[0]) * 100, 2)}%")
    
# print("Saving weights")
# with open('model_parameters.pkl', 'wb') as f:
#     pickle.dump({
#         'w_i_h': w_i_h,
#         'w_h_o': w_h_o,
#         'b_i_h': b_i_h,
#         'b_h_o': b_h_o
#     }, f)
#     f.close()
    
with open('model_parameters.pkl', 'rb') as f:
    
    w_i_h, w_h_o, b_i_h, b_h_o = dict(pickle.load(f)).values()
    f.close()
# print(params)

y_test_labels = np.argmax(y_test_labels, axis=1)
predictions = []
for i, (image, label) in enumerate(zip(x_test_images, y_test_labels)):
    image = image.reshape(784, 1)
    
    # Forward pass: input to hidden
    hidden_pre_activation = (w_i_h @ image) + b_i_h
    hidden_activated = sigmoid(hidden_pre_activation)
        
    # Forward pass: hidden to output
    output_pre_activation = (w_h_o @ hidden_activated) + b_h_o
    output_activated = sigmoid(output_pre_activation)
    
    predictions.append(np.argmax(output_activated))
    
    
print(accuracy_score(predictions, y_test_labels))
    

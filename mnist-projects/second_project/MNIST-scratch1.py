'''
    I will modify the data reading method in the future, so the torch library wont't be necessary
    TODO:
        1, debugging -> still works faulty
        2, read dataset from csv, drop torch lib
        3, document every action
'''

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import torch

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

training_data = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_data = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

training_load = DataLoader(dataset=training_data, batch_size=64, shuffle=True)
test_load = DataLoader(dataset=test_data, batch_size=64, shuffle=False)


class MNISTNeuralNetwork:
    
    def __init__(self, input_nodes, hidden_nodes_1, hidden_nodes_2, output_nodes, lr = 1):
        self.input_nodes = input_nodes
        self.hidden_nodes_1 = hidden_nodes_1
        self.hidden_nodes_2 = hidden_nodes_2
        self.output_nodes  = output_nodes
        self.lr = lr
        
        self.input_to_hidden_1_weights = np.random.randn(input_nodes, hidden_nodes_1)
        self.hidden_1_to_hidden_2_weights = np.random.randn(hidden_nodes_1, hidden_nodes_2)
        self.hidden_2_to_output_weights = np.random.randn(hidden_nodes_2, output_nodes)
        
        self.hidden_1_biases = np.random.randn(hidden_nodes_1)
        self.hidden_2_biases = np.random.randn(hidden_nodes_2)
        self.output_biases = np.random.randn(output_nodes)
        
        
    def relu(self, layer_values):
        return np.maximum(0, layer_values)
    
    def relu_derivative(self, layer_values):
        return np.where(layer_values > 0, 1, 0)
    
    def one_hot_encode(self, labels, num_classes):
        one_hot = torch.zeros(labels.size(0), num_classes)
        one_hot.scatter_(1, labels.unsqueeze(1), 1)
        return one_hot

        
    def forward(self, input_layer_values):
        # input_layer_values = input_layer_values.reshape(64, -1)
        
        input_layer_values = input_layer_values.reshape(input_layer_values.shape[0], -1)
        
        self.hidden_1_values = np.dot(input_layer_values, self.input_to_hidden_1_weights) + self.hidden_1_biases
        self.hidden_1_values = self.relu(self.hidden_1_values)
        
        self.hidden_2_values = np.dot(self.hidden_1_values, self.hidden_1_to_hidden_2_weights) + self.hidden_2_biases
        self.hidden_2_values = self.relu(self.hidden_2_values)
        
        self.output_values = np.dot(self.hidden_2_values, self.hidden_2_to_output_weights) + self.output_biases
        self.output_values = self.relu(self.output_values)
        
        return self.output_values
    
    def backward(self, actual_output, expected_output, input_layer):
        
        if expected_output.dim() == 1: 
            expected_output = self.one_hot_encode(expected_output, num_classes=self.output_nodes)
        
        expected_output = expected_output.cpu().detach().numpy()
        input_layer = input_layer.cpu().detach().numpy()
        
        if input_layer.ndim == 4:
            input_layer = input_layer.reshape(input_layer.shape[0], -1)
        
        error = actual_output - expected_output
        
        self.grad_output = error * self.relu_derivative(actual_output)
        self.grad_hidden_2 = self.grad_output.dot(self.hidden_2_to_output_weights.T) * self.relu_derivative(self.hidden_2_values)
        self.grad_hidden_1 = self.grad_hidden_2.dot(self.hidden_1_to_hidden_2_weights.T) * self.relu_derivative(self.hidden_1_values)
        
        self.output_biases -= self.lr * np.mean(self.grad_output, axis=0)
        self.hidden_1_biases -= self.lr * np.mean(self.grad_hidden_1, axis=0)
        self.hidden_2_biases -= self.lr * np.mean(self.grad_hidden_2, axis=0)
        
        self.hidden_2_to_output_weights -= self.lr * np.dot(self.hidden_2_values.T, self.grad_output)
        self.hidden_1_to_hidden_2_weights -= self.lr * np.dot(self.hidden_1_values.T, self.grad_hidden_2)
        self.input_to_hidden_1_weights -= self.lr * np.dot(input_layer.T, self.grad_hidden_1)
        
        
    def train(self, epochs, training_data):
        for epoch in range(epochs):
            loss_per_epoch = 0.00
            for input, expected in training_data:
                if expected.dim() == 1:
                    expected = self.one_hot_encode(expected, num_classes=self.output_nodes)
                
                output = self.forward(input)
                self.backward(actual_output=output, expected_output=expected, input_layer=input)
                loss_per_epoch += np.sum((expected.cpu().detach().numpy() - output) ** 2)
            print(f"Loss in epoch {epoch}: {loss_per_epoch / len(training_data):.4f}")
                

epochs = 3
learning_rate = 0.001
input_nodes = 28*28
hidden_1 = 128
hidden_2 = 64
output_nodes = 10
correct = 0

model = MNISTNeuralNetwork(input_nodes=input_nodes, hidden_nodes_1=hidden_1, hidden_nodes_2=hidden_2, output_nodes=output_nodes, lr = learning_rate)
model.train(epochs=epochs, training_data=training_load)

for image, label in test_load:
    output = model.forward(image)
    predicted_label = torch.argmax(output, dim=1)
    correct += (predicted_label == label).sum().item()
        
print(f"Accuracy of our neural network is: {100 * correct / len(test_load):.2f}%")   
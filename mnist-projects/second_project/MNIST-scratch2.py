import numpy as np
import requests
import io
import math

'''
    Version two of MNIST NN, gives bad result, still under development
'''

def convert(url):
    resp = requests.get(url)
    data = np.genfromtxt(io.StringIO(resp.text), delimiter=",", skip_header=1)
    images = data[:, :-1] / 255.0
    labels = data[:, -1]
    return images, labels

class MNISTNN:
    
    def __init__(self, input_layer_nodes, hidden_1_nodes, hidden_2_nodes, output_nodes, lr = 1):
        self.input_layer_nodes = input_layer_nodes
        self.hidden_1_nodes = hidden_1_nodes
        self.hidden_2_nodes = hidden_2_nodes
        self.output_nodes = output_nodes
        self.lr = lr
        
        self.input_to_hidden_1_weights = np.random.randn(self.input_layer_nodes, self.hidden_1_nodes)
        self.hidden_1_to_hidden_2_weights = np.random.randn(self.hidden_1_nodes, self.hidden_2_nodes)
        self.hidden_2_to_output_weights = np.random.randn(self.hidden_2_nodes, self.output_nodes)
        
        self.hidden_1_biases = np.random.randn(self.hidden_1_nodes)
        self.hidden_2_biases = np.random.randn(self.hidden_2_nodes)
        self.output_biases = np.random.randn(self.output_nodes)
        
    
    def sigmoid(self, layer_values):
        if isinstance(layer_values, int):
            return 1 / (1 + np.exp(-layer_values))
        return [1 / (1 + np.exp(-x)) for x in range(len(layer_values))]
    
    def sigmoid_derivative(self, layer_values):
        return [self.sigmoid(x) * (1 - self.sigmoid(x)) for x in range(len(layer_values))]
    
    def cross_entropy_loss(self, predicted, expected):
        if isinstance(expected, np.float64) or isinstance(predicted, np.float64):
            expected = np.array([expected])
            predicted = np.array([predicted])
        eps = 1e-10
        loss = -np.sum(expected * np.log(predicted + eps))
        return loss
    
    def forward(self, input_layer_values):
        self.hidden_1_values = np.dot(input_layer_values, self.input_to_hidden_1_weights)
        self.hidden_1_values = self.sigmoid(self.hidden_1_values)
        
        self.hidden_2_values = np.dot(self.hidden_1_values, self.hidden_1_to_hidden_2_weights)
        self.hidden_2_values = self.sigmoid(self.hidden_2_values)
        
        self.output_values = np.dot(self.hidden_2_values, self.hidden_2_to_output_weights)
        self.output_values = self.sigmoid(self.output_values)
        
        return self.output_values
    
    def backward(self, predicted, expected, input_layer_values):
        self.loss = self.cross_entropy_loss(predicted, expected)
    

        self.grad_output = predicted - expected
        self.grad_hidden_2 = self.grad_output.dot(self.hidden_2_to_output_weights.T) * self.sigmoid_derivative(self.hidden_2_values)
        self.grad_hidden_1 = self.grad_hidden_2.dot(self.hidden_1_to_hidden_2_weights.T) * self.sigmoid_derivative(self.hidden_1_values)
        
        self.output_biases -= self.lr * self.grad_output
        self.hidden_2_biases -= self.lr * self.grad_hidden_2
        self.hidden_1_biases -= self.lr * self.grad_hidden_1
        
        self.hidden_2_to_output_weights -= self.lr * np.outer(self.hidden_2_values, self.grad_output)
        self.hidden_1_to_hidden_2_weights -= self.lr * np.outer(self.hidden_1_values, self.grad_hidden_2)
        self.input_to_hidden_1_weights -= self.lr * np.outer(input_layer_values, self.grad_hidden_1)
        
    def train(self, epochs, training_data, iterations):
        for epoch in range(epochs):
            loss_per_epoch = 0.00
            i = 0
            for input, label in training_data:
                predicted = self.forward(input)
                self.backward(predicted, label, input)
                loss_per_epoch += self.cross_entropy_loss(predicted, label)
                i += 1
                if i == iterations:
                    break
            print(f'Loss at epoch {epoch}: {loss_per_epoch / iterations:.4f}')

epochs = 50
learning_rate = 0.0001
input_nodes = 28*28
hidden_1 = 128
hidden_2 = 64
output_nodes = 10
correct = 0
iterations = 10000
test_it = 5000

model = MNISTNN(input_layer_nodes=input_nodes, hidden_1_nodes=hidden_1, hidden_2_nodes=hidden_2, output_nodes=output_nodes, lr=learning_rate)

mnist_url: str = "https://www.openml.org/data/get_csv/52667/mnist_784.arff"

images, labels = convert(mnist_url)

training_data = zip(images, labels)

model.train(epochs=epochs, training_data=training_data, iterations=iterations)

i = 0
for image, label in zip(images, labels):
    output = model.forward(image)
    predicted = np.argmax(output)
    if predicted == label:
        correct += 1
    i += 1
    if i == test_it:
        break
    
    
print(f'NN accuracy: {100 * correct / test_it}%')
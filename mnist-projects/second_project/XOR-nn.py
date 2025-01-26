import numpy as np

class XORNeuralNetwork:
    
    def __init__(self, input_size, hidden_size, output_size, lr):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr
        
        self.input_to_hidden_weights = np.random.randn(input_size, hidden_size)
        self.hidden_to_output_weights = np.random.randn(hidden_size, output_size)
        self.hidden_biases = np.random.randn(hidden_size)
        self.output_bias = np.random.randn(output_size)
        
        
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def forward(self, x):
        
        self.input_to_hidden_layer = np.dot(x, self.input_to_hidden_weights) + self.hidden_biases
        self.input_to_hidden_layer = self.relu(self.input_to_hidden_layer)
        
        self.hidden_to_output_layer = np.dot(self.input_to_hidden_layer, self.hidden_to_output_weights) + self.output_bias
        
        return self.relu(self.hidden_to_output_layer)
    
    def backward(self, input_data, expected_output, output):
        
        error = expected_output - output
        grad_output = error * self.relu_derivative(output)
        
        hidden_error = grad_output.dot(self.hidden_to_output_weights.T) * self.relu_derivative(self.input_to_hidden_layer)
        
        self.hidden_to_output_weights -= self.lr * np.outer(self.input_to_hidden_layer, grad_output)
        self.output_bias -= self.lr * grad_output
        
        self.input_to_hidden_weights -= self.lr * np.outer(input_data, hidden_error)
        self.hidden_biases -= self.lr * hidden_error
        
    def train(self, training_data, epochs):
        for epoch in range(epochs):
            loss_per_epoch = 0.00
            for input_data, expected in training_data:
                output = self.forward(input_data)
                
                self.backward(input_data=input_data, expected_output=expected, output=output)
                
                loss_per_epoch += np.sum((expected - output) ** 2)
            
            if epoch % 1000 == 0:
                print(f"Total loss in epoch {epoch}: {loss_per_epoch / len(training_data):.4f}")
                
                
        
training_data = [
    (np.array([0, 0]), np.array([0])),
    (np.array([0, 1]), np.array([1])),
    (np.array([1, 0]), np.array([1])),
    (np.array([1, 1]), np.array([0])),
]

input_size = 2
hidden_size = 2
output_size = 1
learning_rate = 0.1
epochs = 20000


model = XORNeuralNetwork(input_size=input_size, hidden_size=hidden_size, output_size=output_size, lr=learning_rate)

model.train(training_data=training_data, epochs=epochs)

correct = 0

for input_data, expected in training_data:
    output = model.forward(input_data)
    
    current = 1 if output > 0.5 else 0
    
    if current == expected:
        correct += 1
        
print(f"Accuracy for our neural network at {epochs} epoch is: {100 * correct/len(training_data):.4}%") 
'''
    Building a XOR neural network only using the numpy python module.
    XOR gives a true output, if the input digits are different -> 0 && 1 OR 1 && 0
'''

import numpy as np

class XORNeuralNetwork:

    '''
        Initialization of the network inputs.
        This xor network only needs one of each layer.
        The input layer will contain two nodes (0 or 1)
        Same goes for the hidden layer, two nodes, but we can change it in the future
        And one predicted output value corresponding to the xor function
        
        We also initialize the random weights between the layers, as well as the biases for the hidden-, and output layer's nodes
    '''
        
    def __init__(self, input_nodes, hidden_nodes, output_nodes, lr = 1):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.lr = lr
        
        self.input_to_hidden_weights = np.random.randn(input_nodes, hidden_nodes)
        self.hidden_to_output_weights = np.random.randn(hidden_nodes, output_nodes)
        
        self.hidden_biases = np.random.randn(hidden_nodes)
        self.output_biases = np.random.randn(output_nodes)
        
    '''
        Initializing the activation functions.
        When moving forward in the ANN, we will need an activation function in order to keep the value positive.
        If the value is negative, the function will return 0
        
        We will also need it's derivative form, when propagating backwards.
    '''
        
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    
    '''
        The forward method takes the input layer's values first, then moving it through the hidden layer(s) to the output layer
        np.dot() multiplies two matrices together, returning the next layer's nodes value's. Than we add the biases to the new matrix
        Then we use the activation function on the (values of the) new matrix
        Lastly, we repeat these exact same steps for the output node values
    '''
    
    def forward(self, x):
        
        self.hidden_node_values = np.dot(x, self.input_to_hidden_weights) + self.hidden_biases
        
        self.hidden_node_values = self.relu(self.hidden_node_values)
        
        self.output = np.dot(self.hidden_node_values, self.hidden_to_output_weights) + self.output_biases
        
        return self.relu(self.output)
    
    
    '''
        The backward method first calculates the loss and it's gradient
        We get the gradient of the output by using the relu_derivative function (implemented above)
        Than we transpose the matrix containing the weights from the hidden layer to the output layer
        After that we use the dot() function between the output matrix and the transposed matrix, and by multiplying with the hidden values relu_derivative()
        we get the hidden layer nodes gradient values
        
        All thats left is to update the weights and biases using the np.outer() method, which multiplies all elements with all elements in two matrices
    '''
    
    def backward(self, input_data, output_value, true_value):
        
        error = output_value - true_value
        
        grad_output = error * self.relu_derivative(output_value)
        
        self.grad_hidden = grad_output.dot(self.hidden_to_output_weights.T) * self.relu_derivative(self.hidden_node_values)
    
        self.hidden_to_output_weights -= self.lr * np.outer(self.hidden_node_values, grad_output)
        self.output_biases -= self.lr * grad_output
        
        self.input_to_hidden_weights -= self.lr * np.outer(input_data, self.grad_hidden)
        self.hidden_biases -= self.lr * self.grad_hidden
        
    
    '''
        Now we just train our data a certain amount of times using the functions implemented above
    '''
        
    def train(self, epochs, training_data):
        for epoch in range(epochs):
            loss_per_epoch = 0.00
            for input, expected in training_data:
                output = self.forward(input)
                
                self.backward(input_data=input, output_value=output, true_value=expected)
                
                loss_per_epoch += np.sum((expected - output) ** 2)
                
            if epoch % 1000 == 0:
                print(f"Epoch: {epoch}, Loss: {loss_per_epoch / len(training_data):.3f}")
        

'''
    I implemented the testing part below
    We initialize the training data, testing data and the hyperparameters
    Finally, we train our model on the given data, and test the accuracy
'''

training_data = [
    (np.array([0, 0]), np.array([0])),
    (np.array([0, 1]), np.array([1])),
    (np.array([1, 0]), np.array([1])),
    (np.array([1, 1]), np.array([0])),
    (np.array([0, 0]), np.array([0])),
    (np.array([0, 1]), np.array([1])),
    (np.array([1, 0]), np.array([1])),
    (np.array([1, 1]), np.array([0])),
    (np.array([0, 0]), np.array([0])),
    (np.array([0, 1]), np.array([1])),
    (np.array([1, 0]), np.array([1])),
    (np.array([1, 1]), np.array([0])),
    (np.array([0, 0]), np.array([0])),
    (np.array([0, 1]), np.array([1])),
    (np.array([1, 0]), np.array([1])),
    (np.array([1, 1]), np.array([0])),
    (np.array([0, 0]), np.array([0])),
    (np.array([0, 1]), np.array([1])),
    (np.array([1, 0]), np.array([1])),
    (np.array([1, 1]), np.array([0]))
]    

learning_rate = 0.01
epochs = 20000
correct = 0

model = XORNeuralNetwork(input_nodes=2, hidden_nodes=2, output_nodes=1, lr=learning_rate)

model.train(epochs=epochs, training_data=training_data)

test_data = [
    (np.array([0, 0]), np.array([0])),
    (np.array([0, 1]), np.array([1])),
    (np.array([1, 0]), np.array([1])),
    (np.array([1, 1]), np.array([0])),
]

for input, expected in test_data:
    output = model.forward(input)
    
    predicted = 1 if output > 0.5 else 0
    
    if predicted == expected:
        correct += 1
        
print(f"The accuracy of the XOR neural network is: {100 * correct / len(test_data):.4f}%")
import random

"""
Initializing training data with correct solutions
"""

train_data = [[0,1], [1,0], [1,1], [0,0], [0,1], [1,0], [1,1], [0,0], [0,1], [1,0], [1,1], [0,0], [0,1], [1,0], [1,1], [0,0], [0,1], [1,0], [1,1], [0,0], [0,1], [1,0], [1,1], [0,0]]
train_solution = [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0]

"""
Setting up random weights and biases between the layers. 
This network will user two nodes as input layer, two nodes as hidden layer and one node as output layer.
"""

input_to_hidden_weights = [[random.uniform(-1, 1) for _ in range(2)] for _ in range(2)]
hidden_to_output_weights = [random.uniform(-1, 1) for _ in range(2)]

hidden_biases = [random.uniform(0, 1) for _ in range(2)]
output_bias = random.uniform(0, 1)

"""
Defining the activation function. This time I'll user ReLU and it's derivate version which we will need later in the training process.
"""

def relu(node_value):
    if node_value < 0:
        return 0
    return node_value

def relu_derivative(node_value):
    if node_value <= 0:
        return 0
    return 1

def error(output, expected):
    return (expected - output) ** 2

def gradient_error(output, expected):
    return (-2 * (expected - output) * relu_derivative(output))

"""
Declaring the hyperparameters.
For learning rate I choose 0.01, we will need it later while backpropagating.
For number of training iterations I choose 20000 for better results.
"""

learning_rate = 0.01
epochs = 100000

"""
Training for loop.
"""


for epoch in range(epochs):
    loss_per_epoch = 0.00
    for i, current_input in enumerate(train_data):
        
        hidden_node_value_1 = (current_input[0] * input_to_hidden_weights[0][0] + current_input[1] * input_to_hidden_weights[0][1]) + hidden_biases[0]
        hidden_node_value_2 = (current_input[0] * input_to_hidden_weights[1][0] + current_input[1] * input_to_hidden_weights[1][1]) + hidden_biases[1]
        
        hidden_node_value_1 = relu(hidden_node_value_1)
        hidden_node_value_2 = relu(hidden_node_value_2)
        
        output_value = (hidden_node_value_1 * hidden_to_output_weights[0] + hidden_node_value_2 * hidden_to_output_weights[1]) + output_bias
        
        output_value = relu(output_value)
        
        loss = error(output=output_value, expected=train_solution[i])
        
        loss_per_epoch += loss
        
        output_gradient = gradient_error(output=output_value, expected=train_solution[i])
        
        grad_hidden_to_output_weight_1 = output_gradient * hidden_node_value_1
        grad_hidden_to_output_weight_2 = output_gradient * hidden_node_value_2
        
        grad_output_bias = output_gradient
        
        grad_hidden_1 = output_gradient * hidden_to_output_weights[0] * relu_derivative(hidden_node_value_1)
        grad_hidden_2 = output_gradient * hidden_to_output_weights[1] * relu_derivative(hidden_node_value_2)
        
        grad_inp_weight_1 = grad_hidden_1 * current_input[0]
        grad_inp_weight_2 = grad_hidden_1 * current_input[1]
        grad_inp_weight_3 = grad_hidden_2 * current_input[0]
        grad_inp_weight_4 = grad_hidden_2 * current_input[1]
        
        grad_hidden_bias_1 = grad_hidden_1
        grad_hidden_bias_2 = grad_hidden_2
        
        hidden_to_output_weights[0] -= learning_rate * grad_hidden_to_output_weight_1
        hidden_to_output_weights[1] -= learning_rate * grad_hidden_to_output_weight_2
        
        output_bias -= learning_rate * grad_output_bias
        hidden_biases[0] -= learning_rate * grad_hidden_bias_1
        hidden_biases[1] -= learning_rate * grad_hidden_bias_2
        
        input_to_hidden_weights[0][0] -= learning_rate * grad_inp_weight_1
        input_to_hidden_weights[0][1] -= learning_rate * grad_inp_weight_2
        input_to_hidden_weights[1][0] -= learning_rate * grad_inp_weight_3
        input_to_hidden_weights[1][1] -= learning_rate * grad_inp_weight_4
        


"""
Testing the trained neural network with test data.
"""


test_data = [[0,1], [1,0], [1,1], [0,0]]
test_solution = [1, 1, 0, 0]

correct = 0

for i, current_input in enumerate(test_data):

    hidden_value_1 = (current_input[0] * input_to_hidden_weights[0][0] + current_input[1] * input_to_hidden_weights[0][1]) + hidden_biases[0]
    hidden_value_2 = (current_input[0] * input_to_hidden_weights[1][0] + current_input[1] * input_to_hidden_weights[1][1]) + hidden_biases[1]
    
    hidden_value_1 = relu(hidden_value_1)
    hidden_value_2 = relu(hidden_value_2)
    
    output_value = (hidden_value_1 * hidden_to_output_weights[0] + hidden_value_2 * hidden_to_output_weights[1]) + output_bias
    
    output_value = relu(output_value)
    
    print(f"Before transformation (prediction): {output_value}")
    
    predicted = 1 if output_value > 0.5 else 0
    
    print(f"Prediction: {predicted}, actual: {test_solution[i]}")
    
    if predicted == test_solution[i]:
        correct += 1
        
print(f"Accuracy: {100 * correct / len(test_solution):.3f}%")
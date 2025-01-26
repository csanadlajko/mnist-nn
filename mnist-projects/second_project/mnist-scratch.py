import random


input_data = [[1,0],
              [0,0], 
              [1,1], 
              [0,1]]

def relu(node_value):
    if node_value < 0:
        return 0
    return node_value

def relu_derivative(x):
    return 1 if x > 0 else 0

expected_output = [1, 0, 0, 1]
      
learning_rate = 0.01
epochs = 20000

input_to_hidden_weights = [[random.uniform(-1, 1) for _ in range(2)] for _ in range(2)]
hidden_to_output_weights = [random.uniform(-1, 1) for _ in range(2)]

hidden_bias_values = [random.uniform(0, 1) for _ in range(2)]
output_bias_value = [random.uniform(0, 1)]





for epoch in range(epochs):
    loss_per_epoch = 0.0
    for i, current_input in enumerate(input_data):

        
        hidden_node_1 = (current_input[0] * input_to_hidden_weights[0][0] + current_input[1] * input_to_hidden_weights[0][1]) + hidden_bias_values[0]
        hidden_node_2 = (current_input[0] * input_to_hidden_weights[1][0] + current_input[1] * input_to_hidden_weights[1][1]) + hidden_bias_values[1]
        
        hidden_node_1 = relu(hidden_node_1)
        hidden_node_2 = relu(hidden_node_2)
        
        output = (hidden_node_1 * hidden_to_output_weights[0] + hidden_node_2 * hidden_to_output_weights[1]) + output_bias_value[0]
        
        predicted = relu(output)
        
        error = (expected_output[i] - predicted) ** 2
        
        loss_per_epoch += error
        
        ## backpropagate
        
        ## error gradiense
        grad_output = (-2 * (expected_output[i] - predicted)) * relu_derivative(output)
        
        update_hidden_weight_1 = grad_output * hidden_node_1
        update_hidden_weight_2 = grad_output * hidden_node_2
        
        grad_output_bias = grad_output
        
        grad_hidden_value_1 = grad_output * hidden_to_output_weights[0] * relu_derivative(hidden_node_1)
        grad_hidden_value_2 = grad_output * hidden_to_output_weights[1] * relu_derivative(hidden_node_2)
        
        input_grad__weight_1  = current_input[0] * grad_hidden_value_1
        input_grad__weight_2 = current_input[1] * grad_hidden_value_1
        input_grad__weight_3 = current_input[0] * grad_hidden_value_2
        input_grad__weight_4 = current_input[1] * grad_hidden_value_2
        
        grad_hidden_bias_1 = grad_hidden_value_1
        grad_hidden_bias_2 = grad_hidden_value_2
        
        ## updating weights 
        
        hidden_to_output_weights[0] -= learning_rate * update_hidden_weight_1
        hidden_to_output_weights[1] -= learning_rate * update_hidden_weight_2
        
        output_bias_value[0] -= learning_rate * grad_output_bias
        
        input_to_hidden_weights[0][0] -= learning_rate * input_grad__weight_1
        input_to_hidden_weights[0][1] -= learning_rate * input_grad__weight_2
        input_to_hidden_weights[1][0] -= learning_rate * input_grad__weight_3
        input_to_hidden_weights[1][1] -= learning_rate * input_grad__weight_4
        
        hidden_bias_values[0] -= learning_rate * grad_hidden_bias_1
        hidden_bias_values[1] -= learning_rate * grad_hidden_bias_2
        
    # if epoch % 1000 == 0:
    #     print(f"Loss in the first {epoch} epoch is: {loss_per_epoch}")
        

test_inputs = [[0, 0], [0, 1], [1, 0], [1, 1], [0, 0], [0, 1], [1, 0], [1, 1], [0, 0], [0, 1], [1, 0], [1, 1], [0, 0], [0, 1], [1, 0], [1, 1]]
expected_outputs = [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0]
correct = 0

for i, test_input in enumerate(test_inputs):
    
    hidden_node_1 = relu(test_input[0] * input_to_hidden_weights[0][0] +
                         test_input[1] * input_to_hidden_weights[0][1])
    
    hidden_node_2 = relu(test_input[0] * input_to_hidden_weights[1][0] +
                         test_input[1] * input_to_hidden_weights[1][1])
    
    output = relu(hidden_node_1 * hidden_to_output_weights[0] +
                  hidden_node_2 * hidden_to_output_weights[1])
    
    predicted = 1 if output > 0.5 else 0
    
    if predicted == expected_outputs[i]:
        correct += 1
        
        
print(f"Accuracy of out neural network: {100 * correct / len(expected_outputs):.3f}%")
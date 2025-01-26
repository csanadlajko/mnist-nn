import random


temp_input = [[1,0],
              [0,0], 
              [1,1], 
              [0,1]]

def flatten(matrix):
    result = []
    for item in matrix:
        for values in item:
            result.append(values)
    return result

def relu(value):
    if value < 0:
        return 0
    return value

def get_node_value(input_node, weight, bias):
    return input_node * weight + bias

def init_weights(inputs, outputs):
    for item in inputs:
        for output in outputs:
            weight = random.uniform(-1, 1)
            bias = random.uniform(0, 1)
            output = get_node_value(input_node=item, weight=weight, bias=bias)
            output = relu(output)
        
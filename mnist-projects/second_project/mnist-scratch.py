import numpy as np
import requests
import io

'''
    Converting the files into numpy array using API
    Normalizing the values of the pixels by dividing by 255
    This gives us values between the 0-1 interval
'''

def convert(url):
    resp = requests.get(url=url)
    data = np.genfromtxt(io.StringIO(resp.text), delimiter=",", skip_header=1)
    images = data[:, :-1] / 255.0
    labels = data[:, -1].astype(int)
    return images, labels


class MNISTNeuralNetwork:
    
    '''
        The constructor takes the following parameters as inputs:
            1 - self, the object we are referring to
            2 - input_nodes, number of input neurons ->
                in this case it's 28*28 pixels (an image "flattened" into 1D)
            3 - hidden_1_nodes, the first hidden layer of the NN -> we'll have 128 nodes here
            4 - hidden_2_nodes, the second hidden layer of the NN -> we'll have 64 nodes here
            5 - output_nodes, number of output nodes -> numbers from 0 to 9
            
        We'll also initialize the weights between the layer, and the biases for the hidden and output layers.
        For the initialization of the weights, I used Xavier Initialization, which provides much safer and faster learning.
    '''
    
    def __init__(self, input_nodes, hidden_1_nodes, hidden_2_nodes, output_nodes, lr = 0.01):
        self.input_nodes = input_nodes
        self.hidden_1_nodes = hidden_1_nodes
        self.hidden_2_nodes = hidden_2_nodes
        self.output_nodes = output_nodes
        self.lr = lr
        
        self.input_to_hidden_weights = np.random.randn(self.input_nodes, self.hidden_1_nodes) * self.xavier_initialization(self.input_nodes)
        self.hidden_1_to_2_weights = np.random.randn(self.hidden_1_nodes, self.hidden_2_nodes) * self.xavier_initialization(self.hidden_1_nodes)
        self.hidden_2_to_output_weights = np.random.randn(self.hidden_2_nodes, self.output_nodes) * self.xavier_initialization(self.hidden_2_nodes)
        
        self.hidden_1_biases = np.zeros(self.hidden_1_nodes)
        self.hidden_2_biases = np.zeros(self.hidden_2_nodes)
        self.output_biases = np.zeros(self.output_nodes)
        
    '''
        Moving forward, we initialize the functions we'll throughout the learning process.
        First the activation function, which we'll use on the hidden layers while moving forward.
        We'll also need its derivative form later while performing the backpropagation.
        The Xavier initialization method is explained above.
        With the cross entropy loss function we can calculate and "visualize" how our neural network performs while learning.
        The softmax method converts the output values into probabilities. In other words, it'll be easier for us to determine which number the NN gave as an output.
        Lastly, the one hot encode method gives us the expected number converted into a numpy array.
        For instance, the expected number is 5 -> [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    '''    
    
    def relu(self, layer_nodes):
        return np.maximum(0, layer_nodes)
    
    def relu_derivative(self, layer_nodes):
        return np.where(layer_nodes > 0, 1, 0)
        
    def xavier_initialization(self, layer_nodes):
        return np.sqrt(2. / layer_nodes)
    
    def cross_entropy_loss(self, expected, predicted):
        eps = 1e-10
        return -np.sum(expected * np.log(predicted + eps))
    
    def softmax(self, layer_nodes):
        exp_x = np.exp(layer_nodes - np.max(layer_nodes))
        return exp_x / exp_x.sum(axis=-1, keepdims=True)
    
    def one_hot_encode(self, label):
        expected = np.zeros(self.output_nodes)
        expected[int(label)] = 1
        return expected
    
    '''
        The forward method includes three main steps:
            1, Setting the values of the first hidden layer
            2, Setting the values of the second hidden layer
            3, Setting the values of the output layer
        Throughout our calculations, we user the activation function (relu) on the new values.
        As explained above, we use the softmax method on the output values, so we get the probabilites of each number.
        When propagating forward we use np.dot(), to multiply to matrices together: [the current layer values] * [the weights between the current and the next layer].
    '''
    
    def forward(self, input_layer_values):
        self.hidden_1_values = np.dot(input_layer_values, self.input_to_hidden_weights) + self.hidden_1_biases
        self.hidden_1_values = self.relu(self.hidden_1_values)
        
        self.hidden_2_values = np.dot(self.hidden_1_values, self.hidden_1_to_2_weights) + self.hidden_2_biases
        self.hidden_2_values = self.relu(self.hidden_2_values)
        
        self.output_values = np.dot(self.hidden_2_values, self.hidden_2_to_output_weights) + self.output_biases
        self.output_values = self.softmax(self.output_values)
        
        return self.output_values
        
    def backward(self, expected, predicted, input_values):
        self.loss = self.cross_entropy_loss(expected=expected, predicted=predicted)
        
        grad_output = predicted - expected
        
        grad_hidden_2 = grad_output.dot(self.hidden_2_to_output_weights.T) * self.relu_derivative(self.hidden_2_values)
        grad_hidden_1 = grad_hidden_2.dot(self.hidden_1_to_2_weights.T) * self.relu_derivative(self.hidden_1_values)
        
        self.hidden_2_to_output_weights -= self.lr * np.outer(self.hidden_2_values, grad_output)
        self.hidden_1_to_2_weights -= self.lr * np.outer(self.hidden_1_values, grad_hidden_2)
        self.input_to_hidden_weights -= self.lr * np.outer(input_values, grad_hidden_1)
        
        self.output_biases -= self.lr * grad_output
        self.hidden_2_biases -= self.lr * grad_hidden_2
        self.hidden_1_biases -= self.lr * grad_hidden_1
        
    def train(self, epochs, training_data, num_of_iterations):
        for epoch in range(epochs):
            i = 0
            loss_per_epoch = 0.00
            for image, label in training_data:
                expected = self.one_hot_encode(label=label)
                output = self.forward(image)
                self.backward(expected=expected, predicted=output, input_values=image)
                loss_per_epoch += self.cross_entropy_loss(expected=expected, predicted=output)
                i += 1
                if i == num_of_iterations:
                    break
            print(f"Loss at epoch {epoch + 1}: {loss_per_epoch / num_of_iterations:.4f}")


input_nodes = 28*28
hidden_1 = 128
hidden_2 = 128
output_nodes = 10
epochs = 10
num_of_iterations = 6000
test_iterations = 10000
correct = 0
learning_rate = 0.01
mnist_url = "https://www.openml.org/data/get_csv/52667/mnist_784.arff"

model = MNISTNeuralNetwork(input_nodes=input_nodes, hidden_1_nodes=hidden_1, hidden_2_nodes=hidden_2, output_nodes=output_nodes, lr=learning_rate)


images, labels = convert(mnist_url)

training_data = zip(images, labels)
        
model.train(epochs=epochs, training_data=training_data, num_of_iterations=num_of_iterations)

i = 0
for input, label in training_data:
    predicted = model.forward(input)
    if np.argmax(predicted) == int(label):
        correct += 1
    i += 1
    if i == test_iterations:
        break
    
print(f"Neural network's accuracy: {100 * correct / test_iterations:.4f}%")
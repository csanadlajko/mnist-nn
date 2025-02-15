import numpy as np
import requests
import io

def convert(url):
    """
        Converts the MNIST dataset into a numpy arrays.
        After accessing the data through an API call, we seperate the images and labels,
        which will make the training process much easier.
        We also normalize the images between the values of 0 and 1, so it'll more practical to work with in the future.
    """
    resp = requests.get(url=url)
    data = np.genfromtxt(io.StringIO(resp.text), delimiter=",", skip_header=1)
    images = data[:, :-1] / 255.0
    labels = data[:, -1].astype(int)
    return images, labels


class MNISTNeuralNetwork:
    
    """
        This class contains a solution for an artificial neural network, built from scratch, taught on the MNIST dataset.
        Throughout the learning process, I only used the numpy module, and documented each method thoroughly.
    """
    
    def __init__(self, input_nodes, hidden_1_nodes, hidden_2_nodes, output_nodes, lr = 0.01):
        """
            The constructor takes the following arguements as inputs:
            1 - self, the object we are referring to
            2 - input_nodes, number of input neurons ->
                in this case it's 28*28 pixels (an image "flattened" into 1D)
            3 - hidden_1_nodes, the first hidden layer of the NN -> we'll have 128 nodes here
            4 - hidden_2_nodes, the second hidden layer of the NN -> we'll have 64 nodes here
            5 - output_nodes, number of output nodes -> numbers from 0 to 9
            
            We'll also initialize the weights between the layer, and the biases for the hidden and output layers.
            For the initialization of the weights, I used He Initialization to ensure stable learning.
        """
        self.input_nodes = input_nodes
        self.hidden_1_nodes = hidden_1_nodes
        self.hidden_2_nodes = hidden_2_nodes
        self.output_nodes = output_nodes
        self.lr = lr
        
        self.input_to_hidden_weights = np.random.randn(self.input_nodes, self.hidden_1_nodes) * self.he_initialization(self.input_nodes)
        self.hidden_1_to_2_weights = np.random.randn(self.hidden_1_nodes, self.hidden_2_nodes) * self.he_initialization(self.hidden_1_nodes)
        self.hidden_2_to_output_weights = np.random.randn(self.hidden_2_nodes, self.output_nodes) * self.he_initialization(self.hidden_2_nodes)
        
        self.hidden_1_biases = np.zeros(self.hidden_1_nodes)
        self.hidden_2_biases = np.zeros(self.hidden_2_nodes)
        self.output_biases = np.zeros(self.output_nodes)
        
    
    def relu(self, layer_nodes):
        """
            A simple activation function, which returns 0, if the value's negative
            and returns the value itself, if its positive.
            We only use this function on the hidden layer nodes.
        """
        return np.maximum(0, layer_nodes)
    
    def relu_derivative(self, layer_nodes):
        """
            The derivative form of the previous activation function.
            Similarly, it returns 0, if the value's negative,
            but gives 1 as an output, if value's positive.
            We'll use it to get the gradient's of the hidden layer nodes while backpropagating.
        """
        return np.where(layer_nodes > 0, 1, 0)
        
    def he_initialization(self, layer_nodes):
        """
            This metod is used for better weight initialization.
            It scales the weights based on the number of neurons in the previous layer.
        """
        return np.sqrt(2. / layer_nodes)
    
    def cross_entropy_loss(self, expected, predicted):
        """
            The method takes the expected and predicted output, as parameters.
            With these data, we can calculate and "visualize" how accurate and fast our ANN is performing.
        """
        eps = 1e-10
        return -np.sum(expected * np.log(predicted + eps))
    
    def softmax(self, layer_nodes):
        """
            This method is used on the output layer's values.
            It converts the output values into probabilites,
            so its easier the determine what number our NN gives as an output.
        """
        exp_x = np.exp(layer_nodes - np.max(layer_nodes))
        return exp_x / exp_x.sum(axis=-1, keepdims=True)
    
    def one_hot_encode(self, label):
        """
            The function takes the expected class label as input.
            It returns a one-hot encoded vector, where all elements are 0 except for the index corresponding to the label, which is set to 1.
        """
        expected = np.zeros(self.output_nodes)
        expected[int(label)] = 1
        return expected
    
    def forward(self, input_layer_values):
        """
            The forward method is responsible for propagating the input pixels to the output layer, through the hidden layers.
            We use np.dot(), to multiply matrices together: [the current layer values] * [the weights between the current and the next layer].
            The function returns a vector of probabilties,
            where the highest number is the prediction of the neural network.
        """
        self.hidden_1_values = np.dot(input_layer_values, self.input_to_hidden_weights) + self.hidden_1_biases
        self.hidden_1_values = self.relu(self.hidden_1_values)
        
        self.hidden_2_values = np.dot(self.hidden_1_values, self.hidden_1_to_2_weights) + self.hidden_2_biases
        self.hidden_2_values = self.relu(self.hidden_2_values)
        
        self.output_values = np.dot(self.hidden_2_values, self.hidden_2_to_output_weights) + self.output_biases
        self.output_values = self.softmax(self.output_values)
        
        return self.output_values
        
    def backward(self, expected, predicted, input_values):
        """
            The backward method propagates the data towards the input layer (from the output layer), 
            while updating the weights and biases using gradiant descent.
        """        
        grad_output = predicted - expected
        
        grad_hidden_2 = grad_output.dot(self.hidden_2_to_output_weights.T) * self.relu_derivative(self.hidden_2_values)
        grad_hidden_1 = grad_hidden_2.dot(self.hidden_1_to_2_weights.T) * self.relu_derivative(self.hidden_1_values)
        
        self.hidden_2_to_output_weights -= self.lr * np.outer(self.hidden_2_values, grad_output)
        self.hidden_1_to_2_weights -= self.lr * np.outer(self.hidden_1_values, grad_hidden_2)
        self.input_to_hidden_weights -= self.lr * np.outer(input_values, grad_hidden_1)
        
        self.output_biases -= self.lr * grad_output
        self.hidden_2_biases -= self.lr * grad_hidden_2
        self.hidden_1_biases -= self.lr * grad_hidden_1
        
    def train(self, epochs, training_data):
        """
            Trains the network for a given number of epochs using the training dataset.
            After forwarding an image, we calculate the loss, and propagate backwards, with the functions implemented and documented above.
        """
        for epoch in range(epochs):
            loss_per_epoch = 0.00
            for image, label in training_data:
                expected = self.one_hot_encode(label=label)
                output = self.forward(image)
                self.backward(expected=expected, predicted=output, input_values=image)
                loss_per_epoch += self.cross_entropy_loss(expected=expected, predicted=output)
            print(f"Loss at epoch {epoch + 1}: {loss_per_epoch / len(training_data):.4f}")

"""
    Finally, for the testing part we initialize the hyperparameters first.
    I chose the following parameters, but feel free to change the hidden layer node values in order to earn better results.
    The input and output values are constant in this case,
    as we take a 28*28 picture as an input, and expected values from 0-9 as an output. 
"""

input_nodes = 28*28
hidden_1 = 128
hidden_2 = 64
output_nodes = 10
epochs = 10
test_iterations = 1000
correct = 0
learning_rate = 0.01

"""
    Creating an instance of the neural network and testing it.
    At last, we calculate the accuracy of the neural network based on out test iterations.
    The best I have reached is 96.16%
"""

model = MNISTNeuralNetwork(input_nodes=input_nodes, hidden_1_nodes=hidden_1, hidden_2_nodes=hidden_2, output_nodes=output_nodes, lr=learning_rate)

mnist_url = "https://www.openml.org/data/get_csv/52667/mnist_784.arff"
images, labels = convert(mnist_url)

training_data = list(zip(images, labels))
        
model.train(epochs=epochs, training_data=training_data)

i = 0

for input, label in training_data:
    predicted = model.forward(input)
    if np.argmax(predicted) == int(label):
        correct += 1
    i += 1
    if i == test_iterations:
        break
    
print(f"Neural network's accuracy: {100 * correct / test_iterations:.4f}%")
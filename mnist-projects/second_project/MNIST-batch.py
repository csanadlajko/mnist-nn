import numpy as np
import requests
from io import StringIO

def convert(url):
    req = requests.get(url)
    data = np.genfromtxt(StringIO(req.text), delimiter=",", skip_header=1)
    images = data[:, :-1] / 255.0
    labels = data[:, -1].astype(int)
    return images, labels


class MNISTBatch:
    
    def __init__(self, input_nodes, hidden_1_nodes, hidden_2_nodes, output_nodes, lr = 0.01):
        self.input_nodes = input_nodes
        self.hidden_1_nodes = hidden_1_nodes
        self.hidden_2_nodes = hidden_2_nodes
        self.output_nodes = output_nodes
        self.lr = lr
        
        self.input_to_hidden_weights = np.random.randn(self.input_nodes, self.hidden_1_nodes) * self.he_init(self.input_nodes)
        self.hidden_1_to_2_weights = np.random.randn(self.hidden_1_nodes, self.hidden_2_nodes) * self.he_init(self.hidden_1_nodes)
        self.hidden_2_to_output_weights = np.random.randn(self.hidden_2_nodes, self.output_nodes) * self.he_init(self.hidden_2_nodes)
        
        self.hidden_1_biases = np.zeros(self.hidden_1_nodes)
        self.hidden_2_biases = np.zeros(self.hidden_2_nodes)
        self.output_biases = np.zeros(self.output_nodes)
        
    def he_init(self, prev_nodes):
        return np.sqrt(2. / prev_nodes)
    
    def relu(self, layer_vals):
        return np.maximum(0, layer_vals)
    
    def relu_derivative(self, layer_vals):
        return np.where(layer_vals > 0, 1, 0)
    
    def cross_entropy_loss(self, predicted, expected):
        eps = 1e-10
        return -np.sum(expected * np.log(predicted + eps))
    
    def one_hot_encode(self, batch, label):
        result = np.zeros((len(batch), self.output_nodes))
        result[np.arange(len(batch)), label] = 1
        return result
    
    def softmax(self, output_vals):
        exp_val = np.exp(output_vals - np.max(output_vals, axis=1, keepdims=True))
        return exp_val / np.sum(exp_val, axis=1, keepdims=True)
    
    def forward(self, input_values):
        self.input_values = input_values
        self.hidden_1_values = self.relu(np.dot(self.input_values, self.input_to_hidden_weights) + self.hidden_1_biases)
        self.hidden_2_values = self.relu(np.dot(self.hidden_1_values, self.hidden_1_to_2_weights) + self.hidden_2_biases)
        self.output_values = self.softmax(np.dot(self.hidden_2_values, self.hidden_2_to_output_weights) + self.output_biases)
        return self.output_values
    
    def backward(self, expected):
        
        batch_size = expected.shape[0]
        grad_output = self.output_values - expected
        grad_hidden_2_to_output_weights = np.dot(self.hidden_2_values.T, grad_output) / batch_size
        grad_output_biases = np.mean(grad_output, axis=0)
        
        grad_hidden_2 = np.dot(grad_output, self.hidden_2_to_output_weights.T) * self.relu_derivative(self.hidden_2_values)
        grad_hidden_1 = np.dot(grad_hidden_2, self.hidden_1_to_2_weights.T) * self.relu_derivative(self.hidden_1_values)
        grad_hidden_2_biases = np.mean(grad_hidden_2, axis=0)
        grad_hidden_1_biases = np.mean(grad_hidden_1, axis=0)
        
        grad_hidden_1_to_2_weights = np.dot(self.hidden_1_values.T, grad_hidden_2) / batch_size
        grad_input_to_hidden_weights = np.dot(self.input_values.T, grad_hidden_1) / batch_size
        
        self.hidden_2_to_output_weights -= self.lr * grad_hidden_2_to_output_weights
        self.hidden_1_to_2_weights -= self.lr * grad_hidden_1_to_2_weights
        self.input_to_hidden_weights -= self.lr * grad_input_to_hidden_weights
        self.output_biases -= self.lr * grad_output_biases
        self.hidden_2_biases -= self.lr * grad_hidden_2_biases
        self.hidden_1_biases -= self.lr * grad_hidden_1_biases
        
    def train(self, epochs, training_data, batch_size = 64):
        training_data = np.array(training_data, dtype=object)
        for epoch in range(epochs):
            loss_per_epoch = 0.00
            num_batches = 0
            np.random.shuffle(training_data)
            for i in range(0, len(training_data), batch_size):
                batch = training_data[i:i+batch_size]
                batch_imgs = np.array([sample[0] for sample in batch])
                batch_labels = np.array([sample[1] for sample in batch]).astype(int)
                expected = self.one_hot_encode(batch, batch_labels)
                predicted = self.forward(batch_imgs)
                loss_per_epoch += self.cross_entropy_loss(predicted, expected) / batch_size
                self.backward(expected)
                num_batches += 1
            print(f"Loss at epoch {epoch+1}/{epochs}: {loss_per_epoch / num_batches:.4f}")
    

input_nodes = 28*28
hidden_1_nodes = 128
hidden_2_nodes = 64
output_nodes = 10
learning_rate = 0.01
epochs = 20
batch_size = 64
correct = 0
images, labels = convert("https://www.openml.org/data/get_csv/52667/mnist_784.arff")

model = MNISTBatch(input_nodes, hidden_1_nodes, hidden_2_nodes, output_nodes, learning_rate)

training_data = list(zip(images, labels))

model.train(epochs, training_data, batch_size)

test_it = 1000
i = 0

for image, label in training_data:
    img_batch = np.expand_dims(image, axis=0)
    predicted = model.forward(img_batch)
    if np.argmax(predicted) == int(label):
        correct+=1
    i += 1
    if i == test_it:
        break
    
print(f"Batch MNIST NN accuracy: {correct / test_it * 100:.4f}%")
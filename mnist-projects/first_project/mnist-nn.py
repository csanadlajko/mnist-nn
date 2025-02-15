import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt


transform = transforms.Compose([
    transforms.ToTensor(), ## scale picture to 0-1 value (to a tensor)
    transforms.Normalize((0.5,), (0.5,)) # scale the values around a 0.5 mean, expanding their interval to -1;1 
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
## (down)loading data from MNIST dataset, then creating a three dimensional tensor -> batch size, image height, image width (as seen below)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)


class FirstNeuralNetwork(nn.Module):
    
    def __init__(self):
        super(FirstNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(28*28, 128) ## input to hidden layer
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10) ## hidden layer to output layer
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        ## x.size(i) returns the i. dimension's value (which is 64 in the context -> the batch size), while -1 combines the rest dimension into a one dimension array
        ## in this case its 28*28 -> we get a one dimensional array with the length of 784
        x = torch.relu(self.fc1(x)) ## activation function on hidden layer
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

## hyperparameters used throughput the learning
num_of_iteration = 15
learning_rate = 0.01
momentum = 0.9

model = FirstNeuralNetwork() ## creating an instance of the neural network we implemented above

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum) ## updates the weights, biases while performing backpropagation
criterion = nn.CrossEntropyLoss() ## basically the loss function -> converts the logits to statistic number through softmax function, than uses log loss function to determine the loss value


for epoch in range(num_of_iteration):
    model.train()
    running_loss = 0.0 # the loss value per epoch
    for images, labels in train_loader:
        outputs = model(images)
        
        loss = criterion(outputs, labels) ## calculate the current loss
        optimizer.zero_grad() ## null the gradients
        loss.backward() ## propagate backwards, get new gradient values 
        optimizer.step() ## update the weights and biases
        
        running_loss += loss.item()
        
    print(f"Epoch {epoch+1}/{num_of_iteration}, Loss: {running_loss/len(train_loader):.4f}")
    
    
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
print(f"Test accuracy: {100 * correct / total:.2f}%")



## some visualization
    
model.eval()
examples = 5
correct_examples = []
incorrect_examples = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        for i in range(len(labels)):
            if len(correct_examples) < examples and predicted[i] == labels[i]:
                correct_examples.append((images[i], labels[i], predicted[i]))
            elif len(incorrect_examples) < examples and predicted[i] != labels[i]:
                incorrect_examples.append((images[i], labels[i], predicted[i]))


plt.figure(figsize=(10, 5))
for i, (image, label, pred) in enumerate(correct_examples):
    plt.subplot(2, examples, i + 1)
    plt.imshow(image.squeeze().numpy(), cmap='gray')
    plt.title(f'Original: {label.item()}, NN: {pred.item()}')
    plt.axis('off')


for i, (image, label, pred) in enumerate(incorrect_examples):
    plt.subplot(2, examples, i + 1 + examples)
    plt.imshow(image.squeeze().numpy(), cmap='gray')
    plt.title(f'Original: {label.item()}, NN: {pred.item()}')
    plt.axis('off')

plt.tight_layout()
plt.show()

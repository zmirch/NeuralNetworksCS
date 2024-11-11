import numpy as np
from torchvision.datasets import MNIST
import time

NUM_EPOCHS = 25
LEARNING_RATE = 0.02
LAMBDA_REG = 0.001  # L2 regularization parameter
                    # if = 0, we have std cost func; the bigger -> the more bias, less variance

def download_mnist(is_train: bool):
    dataset = MNIST(root='./data',
                    transform=lambda x: np.array(x).flatten(),
                    download=True,
                    train=is_train)
    mnist_data = []
    mnist_labels = []
    for image, label in dataset:
        mnist_data.append(image)
        mnist_labels.append([int(item == label) for item in range(10)])

    return mnist_data, mnist_labels

def softmax_activation(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Stabilize then exponentiate
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)  # Normalize

def sigmoid_activation(x):
    return 1 / (1 + np.exp(-np.clip(x, -100, 100)))  # Avoid large values for x

def derivative_sigmoid(x):
    return sigmoid_activation(x) * (1 - sigmoid_activation(x))

class NeuralNetwork:
    def __init__(self, input_neurons=784, hidden_neurons=100, output_neurons=10, learning_rate=LEARNING_RATE, lambda_reg=LAMBDA_REG):
        self.a2 = None
        self.z2 = None
        self.a1 = None
        self.z1 = None

        self.w1 = np.random.randn(input_neurons, hidden_neurons)
        self.w2 = np.random.randn(hidden_neurons, output_neurons)

        self.b1 = np.random.randn(hidden_neurons)
        self.b2 = np.random.randn(output_neurons)

        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg

    def forward_propagation(self, x):
        self.z1 = x.dot(self.w1) + self.b1
        self.a1 = sigmoid_activation(self.z1)
        self.z2 = self.a1.dot(self.w2) + self.b2
        self.a2 = softmax_activation(self.z2)
        return self.a2

    def backpropagation(self, x, y):

        # ----Forward pass----
        # Calculate linear combinations of inputs, and activated outputs
        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = sigmoid_activation(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = softmax_activation(self.z2)

        # ----Backward pass----
        delta_output = y - self.a2
        grad_weights_output = np.dot(self.a1.T, delta_output) + self.lambda_reg * self.w2  # applies L2 penalty
        grad_bias_output = np.sum(delta_output, axis=0) # bias gradient

        delta_hidden = np.dot(delta_output, self.w2.T) * derivative_sigmoid(self.z1)
        grad_weights_hidden = np.dot(x.T, delta_hidden) + self.lambda_reg * self.w1  # L2 penalty
        grad_bias_hidden = np.sum(delta_hidden, axis=0)

        self.w1 += self.learning_rate * grad_weights_hidden
        self.b1 += self.learning_rate * grad_bias_hidden
        self.w2 += self.learning_rate * grad_weights_output
        self.b2 += self.learning_rate * grad_bias_output

    def train(self, x, y, epochs=NUM_EPOCHS, batch_size=100):
        for epoch in range(epochs):
            print("-", end="")

        print("")
        for epoch in range(epochs):
            print("*", end="")

            shuffled_indices = np.random.permutation(x.shape[0])
            x_shuffled, y_shuffled = x[shuffled_indices], y[shuffled_indices] # shuffle

            for i in range(0, x.shape[0], batch_size):
                batch_x = x_shuffled[i:i + batch_size]
                batch_y = y_shuffled[i:i + batch_size]
                self.backpropagation(batch_x, batch_y)
        print("")

    def calculate_accuracy(self, x, y):
        predictions = self.forward_propagation(x)
        predicted_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(y, axis=1)
        return np.mean(predicted_labels == true_labels)

if __name__ == '__main__':
    training_data, training_labels = download_mnist(True)
    testing_data, testing_labels = download_mnist(False)

    training_data = np.array(training_data, dtype=np.float64) / 255
    training_labels = np.array(training_labels)
    testing_data = np.array(testing_data, dtype=np.float64) / 255
    testing_labels = np.array(testing_labels)

    MNIST_model = NeuralNetwork(learning_rate=LEARNING_RATE, lambda_reg=LAMBDA_REG)
    begin_time = time.time()
    MNIST_model.train(training_data, training_labels, epochs=NUM_EPOCHS, batch_size=100)
    finish_time = time.time()
    print(f"For {NUM_EPOCHS} epochs:")
    print(f"Training Accuracy: {MNIST_model.calculate_accuracy(training_data, training_labels) * 100}%")
    print(f"Validation Accuracy: {MNIST_model.calculate_accuracy(testing_data, testing_labels) * 100}%")
    print(f"Total duration: {(finish_time - begin_time) / 60} min")

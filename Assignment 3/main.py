import numpy as np
from torchvision.datasets import MNIST
import time

NUM_EPOCHS = 25
LEARNING_RATE = 0.02
LAMBDA_REG = 0.001  # L2 regularization parameter
                    # if = 0, we have standard cost function; the bigger, the more bias, less variance

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

def forward_propagation(x, w1, b1, w2, b2):
    z1 = x.dot(w1) + b1
    a1 = sigmoid_activation(z1)
    z2 = a1.dot(w2) + b2
    a2 = softmax_activation(z2)
    return z1, a1, z2, a2

def backpropagation(x, y, z1, a1, z2, a2, w1, w2, b1, b2, learning_rate, lambda_reg):
    delta_output = y - a2
    grad_weights_output = np.dot(a1.T, delta_output) + lambda_reg * w2  # applies L2 penalty
    grad_bias_output = np.sum(delta_output, axis=0)  # bias gradient

    delta_hidden = np.dot(delta_output, w2.T) * derivative_sigmoid(z1)
    grad_weights_hidden = np.dot(x.T, delta_hidden) + lambda_reg * w1  # L2 penalty
    grad_bias_hidden = np.sum(delta_hidden, axis=0)

    # Update weights and biases
    w1 += learning_rate * grad_weights_hidden
    b1 += learning_rate * grad_bias_hidden
    w2 += learning_rate * grad_weights_output
    b2 += learning_rate * grad_bias_output

    return w1, b1, w2, b2

def calculate_accuracy(x, y, w1, b1, w2, b2):
    _, _, _, predictions = forward_propagation(x, w1, b1, w2, b2)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(y, axis=1)
    return np.mean(predicted_labels == true_labels)

def train(x, y, w1, b1, w2, b2, learning_rate, lambda_reg, epochs=NUM_EPOCHS, batch_size=100):
    for epoch in range(epochs):
        print("-", end="")

    print("")
    for epoch in range(epochs):
        print("*", end="")

        shuffled_indices = np.random.permutation(x.shape[0])
        x_shuffled, y_shuffled = x[shuffled_indices], y[shuffled_indices]  # shuffle

        for i in range(0, x.shape[0], batch_size):
            batch_x = x_shuffled[i:i + batch_size]
            batch_y = y_shuffled[i:i + batch_size]

            # Forward pass
            z1, a1, z2, a2 = forward_propagation(batch_x, w1, b1, w2, b2)

            # Backpropagation and update weights
            w1, b1, w2, b2 = backpropagation(batch_x, batch_y, z1, a1, z2, a2, w1, w2, b1, b2, learning_rate, lambda_reg)
    print("")

if __name__ == '__main__':
    training_data, training_labels = download_mnist(True)
    testing_data, testing_labels = download_mnist(False)

    training_data = np.array(training_data, dtype=np.float64) / 255
    training_labels = np.array(training_labels)
    testing_data = np.array(testing_data, dtype=np.float64) / 255
    testing_labels = np.array(testing_labels)

    input_neurons = 784
    hidden_neurons = 100
    output_neurons = 10

    w1 = np.random.randn(input_neurons, hidden_neurons) * np.sqrt(2 / (input_neurons + hidden_neurons)) # Xavier initialization
    w2 = np.random.randn(hidden_neurons, output_neurons) * np.sqrt(2 / (hidden_neurons + output_neurons))

    b1 = np.random.randn(hidden_neurons)
    b2 = np.random.randn(output_neurons)

    # Train the model
    begin_time = time.time()
    train(training_data, training_labels, w1, b1, w2, b2, learning_rate=LEARNING_RATE, lambda_reg=LAMBDA_REG, epochs=NUM_EPOCHS, batch_size=100)
    finish_time = time.time()

    print(f"For {NUM_EPOCHS} epochs:")
    print(f"Training Accuracy: {calculate_accuracy(training_data, training_labels, w1, b1, w2, b2) * 100}%")
    print(f"Validation Accuracy: {calculate_accuracy(testing_data, testing_labels, w1, b1, w2, b2) * 100}%")
    print(f"Total duration: {(finish_time - begin_time) / 60} min")

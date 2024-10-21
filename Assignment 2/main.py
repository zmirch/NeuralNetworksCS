import numpy as np
from torchvision.datasets import MNIST # collection of 28x28 images of handwritten digits (0 through 9) that we'll use to train and test our neural network.

# Function to download and prepare the MNIST dataset (handwritten digits)
def download_mnist(is_train: bool):
    dataset = MNIST(root='./data',
                    transform=lambda x: np.array(x).flatten(), # Convert 28x28 images to flat 784-elem arrays
                    download=True,
                    train=is_train,)
    mnist_data = []     # Stores flattened images
    mnist_labels = []   # Stores one-hot encoded labels
    for image, label in dataset:
        mnist_data.append(image)
        mnist_labels.append([int(item == label) for item in range(10)]) # One-hot encode labels

    return mnist_data, mnist_labels

# Softmax function to convert raw scores into probabilities of each digit
def calculate_softmax(weighted_sum):
    exp_scores = np.power(np.e, weighted_sum, dtype=np.float64)
    return np.divide(exp_scores, exp_scores.sum())

# Train the model on the training data
def train_model(train_X, train_Y, weights, bias):
    print("Training the model...")
    for epoch in range(NUM_EPOCHS):

        # shuffle training data to make sure network doesn't learn in a fixed order, which helps it generalize
        pairs = list(zip(train_X, train_Y))
        np.random.shuffle(pairs)
        train_X, train_Y = zip(*pairs)

        # Splits the train_X data into smaller chunks/batches, each containing 100 images
        for batch_data, batch_labels in zip(np.array_split(train_X, dataset_len // 100), np.array_split(train_Y, dataset_len // 100)):

            # Modify weights and bias of batch using gradient descent which is a method to reduce the error over time
            raw_output = batch_data.dot(weights) + bias  # Forward pass: calculate raw scores
            predicted = np.array([calculate_softmax(score) for score in raw_output])  # Softmax probabilities
            error_diff = 0.01 * (batch_labels - predicted)  # Backward pass: Calculate error (0.01 is the learning rate)

            weight_adjustment = batch_data.T.dot(error_diff)
            bias_adjustment = error_diff.sum(axis=0)
            weights += weight_adjustment
            bias += bias_adjustment

if __name__ == '__main__':
    # Load MNIST data (True for training, False for testing)
    train_X, train_Y = download_mnist(True) # X for data, Y for labels
    test_X, test_Y = download_mnist(False)

    # Get the number of epochs from the user
    NUM_EPOCHS = int(input("Enter the desired number of epochs: "))

    # Normalize data (divide each pixel by 255 so it's normalized to either 0 or 1)
    train_X = np.array(train_X, dtype=np.float64) / 255
    test_X = np.array(test_X, dtype=np.float64) / 255

    # Initialize weights and bias to zero
    weights = np.zeros((784, 10), dtype=np.float64)
    bias = np.zeros(10, dtype=np.float64)

    dataset_len = len(train_X)

    train_model(train_X, train_Y, weights, bias)

    # Test the model after training, to evaluate performance on the test data
    num_correct = 0
    for image, true_label in zip(test_X, test_Y):
        predicted_probs = calculate_softmax(image.dot(weights) + bias)
        if true_label[np.argmax(predicted_probs)] == 1:
            num_correct += 1
    print(f"Model predicted {num_correct} out of {len(test_X)} test samples, \nwith an accuracy of {(num_correct / len(test_X)) * 100:.2f}%")
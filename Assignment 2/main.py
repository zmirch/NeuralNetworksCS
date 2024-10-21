import numpy as np
from torchvision.datasets import MNIST # collection of 28x28 images of handwritten digits (0 through 9) that we'll use to train and test our neural network.

# Settings of the neural network
#NUM_EPOCHS = 150        # Number of times the model will go through the entire dataset
LEARNING_RATE = 0.001   # Learning rate: how much the model adjusts its predictions each time it learns
INPUT_LEN = 784   # Size of the input (28x28 pixels = 784)
OUTPUT_SIZE = 10  # Number of digits to predict (0-9)
BATCH_SIZE = 100  # Number of images for the model to process at once
CLASS_COUNT = 10  # Total number of possible classes (digits 0-9)

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
        mnist_labels.append([int(item == label) for item in range(CLASS_COUNT)]) # One-hot encode labels

    return mnist_data, mnist_labels

# Softmax function to convert raw scores into probabilities of each digit
def calculate_softmax(weighted_sum):
    exp_scores = np.power(np.e, weighted_sum, dtype=np.float64)
    return np.divide(exp_scores, exp_scores.sum())

# Modify weights and bias using gradient descent
def adjust_weights(inputs, actual_labels, current_weights, current_bias):
    raw_output = inputs.dot(current_weights) + current_bias   # Forward pass: calculate raw scores
    predicted_probs = np.array([calculate_softmax(score) for score in raw_output])  # Softmax probabilities

    error_diff = LEARNING_RATE * (actual_labels - predicted_probs)  # Backward pass: Calculate error

    return inputs.T.dot(error_diff), error_diff.sum(axis=0)
# Test the network after training, to evaluate performance on the test data
def check_model_performance(test_data, test_labels, final_weights, final_bias):
    num_correct = 0
    for image, true_label in zip(test_data, test_labels):
        predicted_probs = calculate_softmax(image.dot(final_weights) + final_bias)
        if true_label[np.argmax(predicted_probs)] == 1:
            num_correct += 1
    accuracy = num_correct / len(test_data)
    print(f"Model predicted {num_correct} out of {len(test_data)} test samples, \nwith an accuracy of {accuracy * 100:.2f}%")
    return accuracy

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
    weights = np.zeros((INPUT_LEN, CLASS_COUNT), dtype=np.float64)
    bias = np.zeros(CLASS_COUNT, dtype=np.float64)

    dataset_len = len(train_X)

    # Training Loop
    print("Training the model...")
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch: {epoch + 1}/{NUM_EPOCHS}")

        # shuffle training data to make sure network doesn't learn in a fixed order, which helps it generalize
        pairs = list(zip(train_X, train_Y))
        np.random.shuffle(pairs)
        train_X, train_Y = zip(*pairs)

        # Split data into batches and update weights/bias for each batch
        for batch_data, batch_labels in zip(
            np.array_split(train_X, dataset_len // BATCH_SIZE), # Splits the train_X data into smaller chunks/batches, each containing BATCH_SIZE images
            np.array_split(train_Y, dataset_len // BATCH_SIZE)):

            # Update weights and bias using gradient descent which is a method to reduce the error over time
            weight_adjustment, bias_adjustment = adjust_weights(batch_data, batch_labels, weights, bias)
            weights += weight_adjustment
            bias += bias_adjustment

    # Check model performance on test data
    check_model_performance(test_X, test_Y, weights, bias)
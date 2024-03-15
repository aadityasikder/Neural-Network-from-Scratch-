# Neural Network from Scratch

## Overview

This notebook implements a neural network from scratch using only NumPy, without relying on libraries like TensorFlow or PyTorch. The neural network architecture consists of an input layer with 784 neurons, two hidden layers with 132 and 40 neurons, respectively, and an output layer with 10 neurons. The sigmoid function is used as the activation function for each neuron.

## Features

- **Neural Network Architecture**: The network has an input layer with 784 neurons, two hidden layers with 132 and 40 neurons, and an output layer with 10 neurons.
- **Activation Function**: The sigmoid function is used as the activation function for each neuron.
- **Forward Propagation**: The forward pass computes the weighted sum of inputs and applies the activation function to get the output for each layer.
- **Backpropagation**: The backward pass propagates the error through the layers to compute the gradients of the cost function with respect to weights and biases.
- **Weight and Bias Update**: The weights and biases are updated using gradient descent to minimize the error between predicted and actual outputs.
- **Training**: The network is trained on the MNIST dataset, which consists of handwritten digit images.
- **Evaluation**: The trained model's accuracy is evaluated on both the training and test sets.

## Dependencies

The following libraries are required to run this notebook:

- NumPy
- Pandas
- Matplotlib

## Usage

1. Clone the repository or download the notebook file.
2. Install the required dependencies.
3. Open the notebook in your preferred environment (e.g., Jupyter Notebook, Google Colab).
4. Run the cells to train the neural network and evaluate its performance.





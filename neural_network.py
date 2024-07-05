import numpy as np


class NeuralNetwork:
    def __init__(self, layers, activation='relu', dropout_rate=0.2):
        # if seed is not None:
        #     np.random.seed(seed)
        self.layers = layers
        self.activation_type = activation
        self.dropout_rate = dropout_rate

        # Initialize activation function based on type
        if activation == 'relu':
            self.activation = self.relu
            self.activation_derivative = self.relu_derivative
        elif activation == 'sigmoid':
            self.activation = self.sigmoid
            self.activation_derivative = self.sigmoid_derivative
        else:
            raise ValueError(f"Unknown activation function: {activation}")

        # Initialize weights and biases
        self.weights = [np.random.randn(self.layers[i], self.layers[i + 1]) / np.sqrt(self.layers[i]) for i in
                        range(len(self.layers) - 1)]
        self.biases = [np.zeros((1, self.layers[i + 1])) for i in range(len(self.layers) - 1)]
        self.a = []

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X, training=False):
        self.a = [X]
        for i in range(len(self.weights)):
            net = np.dot(self.a[-1], self.weights[i]) + self.biases[i]
            activated = self.activation(net)
            if training and self.dropout_rate > 0 and i < len(self.weights) - 1:  # Don't apply dropout on output layer
                dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, size=activated.shape) / (
                            1 - self.dropout_rate)
                activated *= dropout_mask
            self.a.append(activated)
        return self.a[-1]

    def predict(self, X):
        output = self.forward(X, training=False)
        return (output > 0.5).astype(int)

    def backward(self, X, y, output):
        deltas = [None] * len(self.weights)
        deltas[-1] = (output - y) * self.activation_derivative(output)

        for i in reversed(range(len(deltas) - 1)):
            deltas[i] = deltas[i + 1].dot(self.weights[i + 1].T) * self.activation_derivative(self.a[i + 1])

        grads_w = [self.a[i].T.dot(deltas[i]) for i in range(len(deltas))]
        grads_b = [np.sum(deltas[i], axis=0, keepdims=True) for i in range(len(deltas))]

        return grads_w, grads_b

    def update_parameters(self, grads_w, grads_b, learning_rate):
        self.weights = [w - learning_rate * dw for w, dw in zip(self.weights, grads_w)]
        self.biases = [b - learning_rate * db for b, db in zip(self.biases, grads_b)]

    def train(self, X_train, y_train, learning_rate, epochs, batch_size=32):
        num_samples = X_train.shape[0]
        for epoch in range(epochs):
            permutation = np.random.permutation(num_samples)
            X_train = X_train[permutation]
            y_train = y_train[permutation]

            for i in range(0, num_samples, batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]

                output = self.forward(X_batch, training=True)
                grads_w, grads_b = self.backward(X_batch, y_batch, output)
                self.update_parameters(grads_w, grads_b, learning_rate)

            if (epoch + 1) % 100 == 0:
                loss = np.mean((y_train - self.forward(X_train)) ** 2)
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f}")

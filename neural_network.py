import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1.0 - sigmoid(x))


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)


def relu(a, alpha=0.01):
    return max(alpha * a, a)


def mse(y_true, y_predicted):
    return 1 / len(y_true) * sum((y_true - y_predicted) ** 2)


def cross_entropy(y_true, y_pred, epsilon=1e-12):
    predictions = np.clip(y_pred, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(y_true * np.log(predictions + 1e-9)) / N
    return ce


class NeuralNetwork:
    def __init__(self, layers, loss_function='cross_entropy', max_iterations=1000,
                 tol=1e-12, learning_rate=0.01):
        np.random.seed(17)
        self.loss_function = loss_function
        self.layers = layers
        self.total_layers = len(layers)
        self.weights = []
        self.bias = []
        self.a = [None] * (self.total_layers - 1)
        self.z = [0] * (self.total_layers - 1)
        self.weight_initialisation()
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tol
        self.output = 0

    def weight_initialisation(self):
        for i in range(0, self.total_layers - 1):
            self.weights.append(np.random.randn(self.layers[i], self.layers[i + 1]) * 2 - 1)
            self.bias.append(np.random.randn(self.layers[i + 1], 1) * 2 - 1)

        self.weights = np.asarray(self.weights)
        self.bias = np.asarray(self.bias)

    def forward_propagation(self, X):
        self.a[0] = X
        for i in range(1, self.total_layers - 1):
            self.z[i] = (self.a[i - 1] @ self.weights[i - 1]) + self.bias[i - 1][:, 0]
            self.a[i] = sigmoid(self.z[i])

        self.output = sigmoid(self.a[-1] @ self.weights[-1])

    def cost_function(self, y):
        C = 1 / len(y) * cross_entropy(y, self.output)
        return C

    def back_propagation(self, X, y):
        m = len(y)
        deltas = [None] * self.total_layers
        deltas[-1] = (self.output - y)

        for i in reversed(range(len(deltas) - 1)):
            deltas[i] = np.multiply(deltas[i + 1] @ self.weights[i].T, sigmoid_derivative(self.a[i]))

        grad = [0] * (self.total_layers - 1)
        for i in range(self.total_layers - 1):
            grad[i] = self.a[i].T @ deltas[i + 1]

        grad_bias = [0] * (self.total_layers - 1)
        for i in range(self.total_layers - 1):
            grad_bias[i] = np.sum(deltas[i + 1].T, axis=1, keepdims=True)

        self.weights = self.weights - (self.learning_rate / m) * np.array(grad)
        self.bias -= (self.learning_rate / m) * np.asarray(grad_bias)

    def fit(self, X, y, plot=False, verbose=False):
        error = 0
        average_error = []
        history = []
        for i in range(self.max_iterations):
            self.forward_propagation(X)
            self.back_propagation(X, y)
            cost = self.cost_function(y)
            if len(average_error) < 5:
                average_error.append(abs(error - cost))
            else:
                average_error.pop(0)
                average_error.append(abs(error - cost))
            if verbose and i % 100 == 0:
                print(sum(average_error) / len(average_error))
            # if len(average_error) >= 5 and sum(average_error) / len(average_error) < self.tolerance:
                # break
                # pass

            history.append(cost)
            error = cost
        if plot:
            plt.plot(range(0, len(history)), history)
            plt.show()

    def predict(self, X):
        a = [None] * (self.total_layers - 1)
        z = [0] * (self.total_layers - 1)
        a[0] = X
        for i in range(1, self.total_layers - 1):
            z[i] = (a[i - 1] @ self.weights[i - 1]) + self.bias[i - 1][:, 0]
            a[i] = sigmoid(z[i])

        return sigmoid(a[-1] @ self.weights[-1])

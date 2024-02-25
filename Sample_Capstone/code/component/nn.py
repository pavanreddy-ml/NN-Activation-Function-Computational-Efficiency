import numpy as np
from activations import *

class DenseNN:
    def __init__(self, layer_sizes, activation, loss):
        self.activation = activation
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        self.initialize()
        self.loss_type = loss
        self.softmax = SoftmaxNumpy()

        if loss == 'mse':
          self.loss = self.mse_loss
        elif loss == 'bce':
          self.loss = self.cross_entropy_loss
        elif loss == 'cce':
          self.loss = self.categorical_cross_entropy_loss

    @staticmethod
    def mse_loss(y_true, y_pred):
        return ((y_true - y_pred) ** 2).mean()

    @staticmethod
    def cross_entropy_loss(y_true, y_pred):
        m = y_true.shape[0]
        return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / m

    @staticmethod
    def categorical_cross_entropy_loss(y_true, y_pred):
      y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
      m = y_true.shape[0]
      return -np.sum(y_true * np.log(y_pred_clipped)) / m

    def initialize(self):
        for i in range(1, len(self.layer_sizes)):
            self.weights.append(np.random.randn(self.layer_sizes[i-1], self.layer_sizes[i]) * 0.01)
            self.biases.append(np.zeros((1, self.layer_sizes[i])))

    def forward_pass(self, X):
        activation = X
        activations = [X]
        zs = []
        grads = []

        for i in range(len(self.weights)):
            z = np.dot(activation, self.weights[i]) + self.biases[i]
            zs.append(z)
            if i == len(self.weights) - 1 and self.loss == 'cce':
                print(1)
                activation, grad = self.softmax.softmax_forward(z)
            else:
                activation, grad = self.activation(z)
            activations.append(activation)
            grads.append(grad)

        return activations, grads, zs

    def backpropagation(self, X, y):
        m = y.shape[0]
        activations, grads, zs = self.forward_pass(X)

        dWs = []
        dBs = []

        if self.loss == 'cce':
            delta = activations[-1] - y
        else:
            delta = (activations[-1] - y) * grads[-1]
        deltas = [delta]

        for l in range(len(self.weights) - 1, -1, -1):
            dW = np.dot(activations[l].T, deltas[-1]) / m
            dB = np.sum(deltas[-1], axis=0, keepdims=True) / m
            dWs.insert(0, dW)
            dBs.insert(0, dB)
            if l > 0:
                delta = np.dot(deltas[-1], self.weights[l].T) * grads[l-1]
                deltas.append(delta)

        return dWs, dBs

    def update_weights(self, dWs, dBs, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * dWs[i]
            self.biases[i] -= learning_rate * dBs[i]

    def train(self, X, y, learning_rate, epochs, batch_size, verbose=False):
        n_samples = X.shape[0]

        for epoch in range(epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                dWs, dBs = self.backpropagation(X_batch, y_batch)
                self.update_weights(dWs, dBs, learning_rate)

            if verbose and epoch % 100 == 0:
                pred = self.forward_pass(X)[0][-1]
                loss = self.get_loss(X, y)
                print(f"Epoch {epoch}, Loss: {loss}")

    def get_loss(self, X, y):
      pred = self.forward_pass(X)[0][-1]
      if self.loss_type == "bce":
          predictions = pred > 0.5
          accuracy = np.mean(predictions == y)
          return accuracy

      elif self.loss_type == "cce":
          predictions = np.argmax(pred, axis=1)
          if y.ndim == pred.ndim and y.shape[1] > 1:
              y = np.argmax(y, axis=1)
          accuracy = np.mean(predictions == y)
          return accuracy

      else:
          return self.loss(y, pred)
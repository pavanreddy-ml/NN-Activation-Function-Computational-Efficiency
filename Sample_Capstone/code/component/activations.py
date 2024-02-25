import numpy as np

class PiecewiseSigmoidNumpy:
    def __init__(self, n_pieces=1024, beta=1, x_min=-10, x_max=10):
        self.x_range = (x_min, x_max)
        self.n_pieces = n_pieces
        self.beta = beta
        self.initialize_values()

    def initialize_values(self):
        x_pieces = np.linspace(self.x_range[0], self.x_range[1], self.n_pieces + 1)
        y_pieces = self.sigmoid(x_pieces)

        self.x = np.ascontiguousarray(x_pieces[:-1])
        self.y = np.ascontiguousarray(y_pieces[:-1])
        slopes = (y_pieces[1:] - y_pieces[:-1]) / (x_pieces[1:] - x_pieces[:-1])

        slopes[0] = 0
        slopes[-1] = 0

        self.slopes = np.ascontiguousarray(slopes)
        self.interval = self.x[1] - self.x[0]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-self.beta * x))

    def forward(self, x):
        segment = ((x - self.x_range[0]) / self.interval).astype(int)
        segment = np.clip(segment, 0, self.n_pieces - 1)
        grads = self.slopes[segment]
        return self.y[segment] + (grads * (x - self.x[segment])), grads


class ContinuousSigmoidNumpy:
    def __init__(self):
        pass

    def forward(self, x):
        y = 1 / (1 + np.exp(-x))
        return y, y * (1 - y)



class PiecewiseTanhNumpy:
    def __init__(self, n_pieces=1024, x_min=-10, x_max=10):
        self.x_range = (x_min, x_max)
        self.n_pieces = n_pieces
        self.initialize_values()

    def initialize_values(self):
        x_pieces = np.linspace(self.x_range[0], self.x_range[1], self.n_pieces + 1)
        y_pieces = self.tanh(x_pieces)

        self.x = np.ascontiguousarray(x_pieces[:-1])
        self.y = np.ascontiguousarray(y_pieces[:-1])
        slopes = (y_pieces[1:] - y_pieces[:-1]) / (x_pieces[1:] - x_pieces[:-1])

        slopes[0] = 0
        slopes[-1] = 0

        self.slopes = np.ascontiguousarray(slopes)
        self.interval = self.x[1] - self.x[0]

    def tanh(self, x):
        return np.tanh(x)

    def forward(self, x):
        segment = ((x - self.x_range[0]) / self.interval).astype(int)
        segment = np.clip(segment, 0, self.n_pieces - 1)
        grads = self.slopes[segment]
        return self.y[segment] + (grads * (x - self.x[segment])), grads


class ContinuousTanhNumpy:
    def __init__(self):
        pass

    def forward(self, x):
        y = np.tanh(x)
        return y, 1 - (y**2)


class SoftmaxNumpy:
    def __init__(self):
        pass

    def softmax_forward(self, x):
        shift_x = x - np.max(x, axis=1, keepdims=True)
        exps = np.exp(shift_x)
        softmax_output = exps / np.sum(exps, axis=1, keepdims=True)
        return softmax_output, None
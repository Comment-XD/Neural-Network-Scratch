import numpy as np
from src.layer import Layer

class Activation(Layer):
    def __init__(self, name:str):

        assert name in activation_forward.keys(), f"Activation {name} not supported"

        self.activation_forward = activation_forward[name]
        self.activation_backward = activation_backward[name]

    def forward(self, x):

        self.input = x
        return self.activation_forward(x)

    def backward(self, x):
        return self.activation_backward(x)

def softmax_backward(x, output_gradient):
    # This version is faster than the one presented in the video
    n = np.size(x)
    return np.dot((np.identity(n) - x.T) * x, output_gradient)

activation_forward = {
    'tanh': lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)),
    'relu': lambda x: np.maximum(0, x),
    'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
    'softmax': lambda x: np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
}

activation_backward = {
    'tanh': lambda x: 1 - x ** 2,
    'relu': lambda x: np.where(x > 0, 1, 0),
    'sigmoid': lambda x: x * (1 - x),
    'softmax': lambda x: 1
}

class Tanh(Activation):
    def __init__(self):
        super().__init__("tanh")

    def forward(self, x):
        return super().forward(x)

    def backward(self, x):
        x = self.forward(x)
        return super().backward(x)

    def __str__(self):
        return "Tanh()"

class Sigmoid(Activation):
    def __init__(self):
        super().__init__("sigmoid")

    def forward(self, x):
        return super().forward(x)

    def backward(self, x):
        x = self.forward(x)
        return super().backward(x)

    def __str__(self):
        return "Sigmoid()"

class ReLU(Activation):
    def __init__(self):
        super().__init__("relu")

    def forward(self, x):
        return super().forward(x)

    def backward(self, x):
        return super().backward(x)

    def __str__(self):
        return "ReLU()"
    
class Softmax(Activation):
    def __init__(self):
        super().__init__("softmax")

    def forward(self, x):
        return super().forward(x)

    def backward(self, x):
        return super().backward(x)

    def __str__(self):
        return "Softmax()"

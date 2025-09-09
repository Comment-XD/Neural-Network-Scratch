import numpy as np
from src.activation import Activation

from typing import Optional
from typing import List, Union
from src.activation import Activation

class Layer:
    def __init__(self):
        self.weights = None
        self.bias = None

    def forward(self):
        pass

    def backward(self):
        pass
    

class Linear(Layer):
    def __init__(self, in_features, out_features, activation:Optional[str]=None):
        super().__init__()
        self.weights = np.random.randn(in_features, out_features)
        # self.bias = np.random.randn(out_features)
        self.activation = Activation(activation) if activation else None

    def forward(self, x):

        self.input = x
        output = np.dot(self.input, self.weights)

        if self.activation:
            output = self.activation.forward(output)

        return output

    def backward(self, gradient, alpha):
        if self.activation:
            a_input = self.forward(self.input)
            gradient = np.multiply(gradient, self.activation.backward(a_input))

        self.weights -= alpha * np.dot(self.input.T, gradient)
        # self.bias -= alpha * np.sum(gradient)

        gradient = np.dot(gradient, self.weights.T)
        return gradient

    def __str__(self):
        return f"Linear(in_feat={self.weights.shape[0]}, out_feat={self.weights.shape[1]})"

class Conv2d(Layer):
    def __init__(self, in_features, out_features, activation:Optional[str]=None):
        super().__init__()
        self.weights = np.random.randn(in_features, out_features)
        # self.bias = np.random.randn(out_features)
        self.activation = Activation(activation) if activation else None

    def forward(self, x):

        self.input = x
        output = np.dot(self.input, self.weights)

        if self.activation:
            output = self.activation.forward(output)

        return output

    def backward(self, gradient, alpha):
        if self.activation:
            a_input = self.forward(self.input)
            gradient = np.multiply(gradient, self.activation.backward(a_input))

        self.weights -= alpha * np.dot(self.input.T, gradient)
        # self.bias -= alpha * np.sum(gradient)

        gradient = np.dot(gradient, self.weights.T)
        return gradient

    def __str__(self):
        return f"Linear(in_feat={self.weights.shape[0]}, out_feat={self.weights.shape[1]})"
    
class Sequential:
    def __init__(self, layers:List[Union[Layer, Activation]]):
        super().__init__()
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, gradient, alpha):
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, alpha)
        return gradient

    def __call__(self, x):
        return self.forward(x)

    def __str__(self):
        model_structure_str = "Sequential(["

        for layer in self.layers:
            model_structure_str += f"{layer}, "

        model_structure_str += "])"

        return model_structure_str
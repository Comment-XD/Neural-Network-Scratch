import numpy as np

class Loss:
    def __call__(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true

    def backward(self):
        pass
    
class MSELoss(Loss):
    def __init__(self, epsilon:float=1e-10):
        super().__init__()
        self.epsilon = epsilon

    def __call__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

        return np.mean(np.square(y_pred - y_true))

    def backward(self):
        return 2 * np.mean(self.y_pred - self.y_true)
    
class BCELoss(Loss):
    def __init__(self, epsilon:float=1e-10):
        super().__init__()
        self.epsilon = epsilon

    def __call__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

        return np.mean(-y_true * np.log(y_pred + self.epsilon) - (1 - y_true) * np.log(1 - y_pred + self.epsilon))

    def backward(self):
        return ((1 - self.y_true) / (1 - self.y_pred) - self.y_true / self.y_pred) / np.size(self.y_true)
    
class CrossEntropyLoss(Loss):
    def __init__(self, epsilon:float=1e-10):
        super().__init__()
        self.epsilon = epsilon

    def __call__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

        return np.mean(-np.dot(y_true, np.log(y_pred + self.epsilon)))

    def backward(self):
        return 1.0
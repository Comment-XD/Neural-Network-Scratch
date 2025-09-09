import numpy as np
from typing import Optional


# Need to improve this using the dataloader system
class Trainer:
    def __init__(self, model, X_train, X_test, y_train, y_test, loss):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        self.loss = loss
        self.loss_chart = []

    def run(self, lr:float=1e-4, verbose:bool=False, epochs:int=100):
        self.epochs = epochs

        for i in range(epochs):

            output = self.model(self.X_train)
            loss = np.mean(self.loss(self.y_train, output), axis=-1)
            self.loss_chart.append(loss)
            
            gradient = self.loss.backward()
            self.model.backward(gradient, lr)

            if verbose:
                if i % (epochs // 10) == 0:
                    print(f"Epoch {i+1}/{epochs}")
                    output = self.model(self.X_test)
                    acc = np.sum(output.argmax(axis=-1) == self.y_test) / self.y_test.shape[0]
                    print(f"| {self.loss.__class__.__name__} -> {loss: .4f} | Test Accuracy: {acc : .2f}")

            
            

class DataLoader(object):
    def __call__(dataset, batch_size, shuffle:bool=True, drop_last:bool=False):
        nbatches = len(dataset) // batch_size
        
        if drop_last:
            pass
        
        return (dataset[i, ] for i in range())

def train_test_split(X, y, train_size:float=0.8, test_size:float=0.2, shuffle:bool=True, random_seed:Optional[int]=None):
    np.random.seed(random_seed)
    
    if shuffle:
        dataset = np.concat([X, y[:, np.newaxis]], axis=1)
        np.random.shuffle(dataset)
        
        X, y = dataset[:, :-1], dataset[:, -1]

    train_idx = int(len(X) * train_size)
    
    X_train, X_test = X[:train_idx], X[train_idx:]
    y_train, y_test = y[:train_idx], y[train_idx:]
    
    return X_train, X_test, y_train, y_test
    


    
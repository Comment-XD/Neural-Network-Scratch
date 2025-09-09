import numpy as np

from src.modules import *
from src.activation import *
from src.loss import *
from src.utils import Trainer, train_test_split

from sklearn.datasets import load_digits

features, labels = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(features, labels)


model = Sequential([
    Linear(64, 100),
    Sigmoid(),
    Linear(100, 100),
    Sigmoid(),
    Linear(100, 10),
    Softmax()
])


trainer = Trainer(model, X_train, X_test, y_train, y_test, loss=CrossEntropyLoss(onehot=True))
trainer.run(lr=3e-4, verbose=True, epochs=100)




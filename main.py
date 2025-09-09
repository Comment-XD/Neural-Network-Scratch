import numpy as np

from src.modules import *
from src.activation import *
from src.utils import Trainer

from sklearn.datasets import load_digits

digits_dataset = load_digits(as_frame=False)["data"] / 255.0

print(digits_dataset.shape)


fc = Sequential([
    Linear(64, 100),
    Sigmoid(),
    Linear(100, 10)
])

# data = np.random.rand(64, 5)

# print(fc(data).shape)
# # Trainer(fc, )

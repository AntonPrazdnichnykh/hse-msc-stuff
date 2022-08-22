import numpy as np
from numpy.random import uniform, normal
from sklearn.datasets import dump_svmlight_file

X = [uniform(-1, 1) * normal() + uniform(-1, 1) for _ in range(1000)]
y = [0 if x < 0 else 1 for x in X]
X = np.array(X).reshape((-1, 1))

dump_svmlight_file(X, y, 'my_dataset.txt', zero_based=False)

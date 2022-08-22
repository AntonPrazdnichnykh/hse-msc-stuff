import numpy as np
from typing import List


class MyMatrix:
    def __init__(self, data: List[List[float]]):
        l = len(data[0])
        for row in data:
            if len(row) != l:
                raise ValueError('Rows of input matrix must be of same length')
        self.shape = (len(data), l)

        self._data = data

    def __getitem__(self, idx):
        if not isinstance(idx, (int, tuple)):
            raise ValueError('Index must be either in or tuple of ints')
        if isinstance(idx, int):
            return self._data[idx]
        if len(idx) > 2:
            raise ValueError('Matrix has no more than 2 indices')
        i, j = idx
        return self._data[i][j]

    def __add__(self, other):
        if other.shape != self.shape:
            raise ValueError('Added arrays must be of same shape')
        return self.__class__(
            [[self[i, j] + other[i, j] for j in range(self.shape[1])] for i in range(self.shape[0])]
        )

    def __mul__(self, other):
        if other.shape != self.shape:
            raise ValueError('Multiplied arrays must be of same shape')
        return self.__class__(
            [[self[i, j] * other[i, j] for j in range(self.shape[1])] for i in range(self.shape[0])]
        )

    def __matmul__(self, other):
        if self.shape[1] != other.shape[0]:
            raise ValueError('Matrices of these shapes cannot be multiplied')

        return self.__class__(
            [
                [sum(self[i, j] * other[j, k] for j in range(self.shape[1])) for k in range(other.shape[1])]
                for i in range(self.shape[0])
            ]
        )

    def __str__(self):
        return '[' + ',\n '.join([str(row) for row in self._data]) + ']'

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self._data == other._data


if __name__ == "__main__":
    np.random.seed(0)
    x_ = np.random.randint(0, 10, (10, 10))
    y_ = np.random.randint(0, 10, (10, 10))
    x = MyMatrix(x_.tolist())
    y = MyMatrix(y_.tolist())

    sum_res = x + y
    assert sum_res._data == (x_ + y_).tolist(), "Wrong sum value"
    with open('artifacts/easy/matrix+.txt', 'w') as f:
        f.write(str(sum_res))

    prod_res = x * y
    assert prod_res._data == (x_ * y_).tolist(), "Wrong sum value"
    with open('artifacts/easy/matrix*.txt', 'w') as f:
        f.write(str(prod_res))

    matmul_res = x @ y
    assert matmul_res._data == (x_ @ y_).tolist(), "Wrong sum value"
    with open('artifacts/easy/matrix@.txt', 'w') as f:
        f.write(str(matmul_res))

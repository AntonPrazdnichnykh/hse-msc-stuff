import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
from numbers import Number


class MyArray(NDArrayOperatorsMixin):
    def __init__(self, data):
        self.data = np.asarray(data)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == '__call__':
            values = []
            for inp in inputs:
                if isinstance(inp, (Number, np.ndarray)):
                    values.append(inp)
                elif isinstance(inp, self.__class__):
                    values.append(inp.data)
                else:
                    raise NotImplementedError()
            return self.__class__(ufunc(*values, **kwargs))
        return NotImplementedError()

    def __str__(self):
        s = [[str(e) for e in row] for row in self.data.tolist()]
        lens = [max(map(len, col)) for col in zip(*s)]
        fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
        table = [fmt.format(*row) for row in s]
        return '\n'.join(table)

    def save(self, save_path):
        with open(save_path, 'w') as f:
            f.write(str(self))


if __name__ == "__main__":
    np.random.seed(0)
    x_ = np.random.randint(0, 10, (10, 10))
    y_ = np.random.randint(0, 10, (10, 10))
    x = MyArray(x_)
    y = MyArray(y_)

    sum_res = x + y
    assert np.array_equal(sum_res.data, x_ + y_), "Wrong sum value"
    sum_res.save('artifacts/medium/matrix+.txt')

    prod_res = x * y
    assert np.array_equal(prod_res.data, x_ * y_), "Wrong prod value"
    prod_res.save('artifacts/medium/matrix*.txt')

    matmul_res = x @ y
    assert np.array_equal(matmul_res.data, x_ @ y_), "Wrong matmul value"
    matmul_res.save('artifacts/medium/matrix@.txt')

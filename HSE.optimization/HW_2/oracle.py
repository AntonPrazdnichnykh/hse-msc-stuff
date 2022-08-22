from sklearn.datasets import load_svmlight_file
import numpy as np
import scipy.sparse as sp
from scipy.special import expit


class Oracle:

    def __init__(self, data_path):
        x, y = load_svmlight_file(data_path, zero_based=False)
        y[y == -1] = 0
        y[y == 2] = 0   # для breast cancer
        y[y == 4] = 1   # для breast cancer
        x = sp.hstack((x, np.ones((x.shape[0], 1))), format='csr')
        self.x = x                      # строки матрицы -- векторы данных
        self.y = y.reshape((-1, 1))     # вектор меток
        self.n = self.y.shape[0]        # количество данных

    def value(self, w):
        z = self.x.dot(w)
        id = np.ones(self.n).reshape(1, -1)
        return ((- self.y.reshape((1, -1)) @ z + id @ np.log(1 + np.exp(z))) / self.n).item()

    def grad(self, w):
        z = self.x.dot(w)
        return self.x.transpose().dot(expit(z) - self.y) / self.n

    def hessian(self, w):
        z = self.x.dot(w)
        M = sp.diags((expit(z) * (1 - expit(z))).reshape((1, -1))[0])
        return (self.x.transpose().dot(M.dot(self.x)) / self.n).toarray()

    def hessian_vec_product(self, w, d):
        z = self.x.dot(w)
        M = sp.diags((np.exp(z) / (1 + np.exp(z)) ** 2).reshape((1, -1))[0])
        return self.x.transpose().dot(M.dot(self.x.dot(d))) / self.n

    def fuse_value_grad(self, w):
        return self.value(w), self.grad(w)

    def fuse_value_grad_hessian(self, w):
        return self.value(w), self.grad(w), self.hessian(w)

    def fuse_value_grad_hessian_vec_product(self, w, d):
        return self.value(w), self.grad(w), self.hessian_vec_product(w, d)


def make_oracle(data_path):
    return Oracle(data_path)


def der(f, w0, h=1e-3):
    n = w0.shape[0]
    return np.array([(f(w0 + h*ei.reshape((-1, 1))) - f(w0 - h*ei.reshape((-1, 1)))) / (2 * h)
                     for ei in np.eye(n)]).reshape((-1, 1))


def der2(f, w0, h=1e-3):
    n = w0.shape[0]
    idt = np.eye(n)
    res = [[(f(w0 + h / 2 * (ei + ej).reshape((-1,1))) - f(w0 + h / 2 * (ej - ei).reshape((-1, 1))) - f(w0 +
            h / 2 * (ei - ej).reshape((-1, 1))) + f(w0 - h / 2 * (ei + ej).reshape((-1, 1)))) / h**2
            for ej in idt] for ei in idt]
    return np.array(res)


if __name__ == '__main__':
    x, y = load_svmlight_file('a1a.txt', zero_based=False)
    x = sp.hstack((x, np.ones((x.shape[0], 1))), format='csr')
    y[y == -1] = 0
    y = sp.csr_matrix(y).reshape((-1, 1))
    n = x[0].shape[1]
    w1 = np.zeros(n).reshape((-1, 1))
    w2 = np.ones(n).reshape((-1, 1))

    oracle = make_oracle('breast-cancer_scale.txt.txt')
    f = lambda w: oracle.value(w)
    # print(np.linalg.norm(oracle.grad(w1) - der(f, w1)))
    print(np.linalg.norm(oracle.hessian(w1) - der2(f, w1)))

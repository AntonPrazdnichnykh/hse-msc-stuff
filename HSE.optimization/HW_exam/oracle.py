from sklearn.datasets import load_svmlight_file
import numpy as np
import scipy.sparse as sp
from scipy.special import expit
from scipy import linalg


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


def make_oracle(data_path, penalty=None, reg=0.1):
    if penalty == 'l1':
        return OracleLasso(data_path, reg)
    if penalty == 'log_barriers':
        return OracleLogBarrier(data_path, reg)
    return Oracle(data_path)


def der(f, w0, h=1e-3):
    n = w0.shape[0]
    return np.array([(f(w0 + h*ei.reshape((-1, 1))) - f(w0 - h*ei.reshape((-1, 1)))) / (2 * h)
                     for ei in np.eye(n)]).reshape((-1, 1))


def der2(f, w0, h=1e-3):
    n = w0.shape[0]
    idt = np.eye(n)
    res = [[(f(w0 + h / 2 * (ei + ej).reshape((-1, 1))) - f(w0 + h / 2 * (ej - ei).reshape((-1, 1))) - f(w0 +
            h / 2 * (ei - ej).reshape((-1, 1))) + f(w0 - h / 2 * (ei + ej).reshape((-1, 1)))) / h**2
            for ej in idt] for ei in idt]
    return np.array(res)


class OracleLasso:
    def __init__(self, data_path, reg=0.001):
        x, y = load_svmlight_file(data_path, zero_based=False)
        y_min, y_max = np.min(y), np.max(y)
        y = (y - y_min) / (y_max - y_min)   # 0,1 скейлинг
        x = sp.hstack((x, np.ones((x.shape[0], 1))), format='csr')
        self.x = x  # строки матрицы -- векторы данных
        self.y = y.reshape((-1, 1))  # вектор меток
        self.n = self.y.shape[0]  # количество данных
        self.reg = reg  # регуляризация

    def value(self, w):
        z = self.x.dot(w)
        id = np.ones(self.n).reshape(1, -1)
        return ((- self.y.reshape((1, -1)) @ z + id @ np.log(1 + np.exp(z))) / self.n +
                self.reg * linalg.norm(w, ord=1)).item()

    def f_value(self, w):
        z = self.x.dot(w)
        id = np.ones(self.n).reshape(1, -1)
        return ((- self.y.reshape((1, -1)) @ z + id @ np.log(1 + np.exp(z))) / self.n).item()

    def h_value(self, w):
        return self.reg * linalg.norm(w, ord=1)

    def grad_f(self, w):
        z = self.x.dot(w)
        return self.x.transpose().dot(expit(z) - self.y) / self.n


class OracleLogBarrier:
    def __init__(self, data_path, reg=0.01):
        x, y = load_svmlight_file(data_path, zero_based=False)
        y_min, y_max = np.min(y), np.max(y)
        y = (y - y_min) / (y_max - y_min)  # 0,1 скейлинг
        x = sp.hstack((x, np.ones((x.shape[0], 1))), format='csr')
        self.x = x  # строки матрицы -- векторы данных
        self.y = y.reshape((-1, 1))  # вектор меток
        self.n = self.y.shape[0]  # количество данных
        self.m = x.shape[1]  # размерность пространства признаков
        self.reg = reg  # регуляризация

    def value(self, t, w_pm):
        m = self.m
        w_plus, w_minus = w_pm[:m], w_pm[m:]
        w = w_plus - w_minus
        z = self.x.dot(w)
        id_n = np.ones(self.n).reshape(1, -1)
        id_m = np.ones((1, m))
        f = (- self.y.reshape((1, -1)) @ z + id_n @ np.log(1 + np.exp(z))) / self.n
        print('val:', np.log(w_plus).shape, np.log(w_minus).shape, np.log(self.reg - w_plus - w_minus).shape)
        return (t * f - id_m @ (np.log(w_plus) + np.log(w_minus) + np.log(self.reg - w_plus - w_minus))).item()

    def grad(self, t, w_pm):
        m = self.m
        w_plus, w_minus = w_pm[:m], w_pm[m:]
        w = w_plus - w_minus
        z = self.x.dot(w)
        grad_F = self.x.transpose().dot(expit(z) - self.y) / self.n
        grad = np.zeros((2 * m, 1))
        grad[:m] = t * grad_F + 1 / (self.reg - w_plus - w_minus) - 1 / w_plus
        grad[m:] = -t * grad_F + 1 / (self.reg - w_plus - w_minus) - 1 / w_minus
        return grad

    def hessian(self, t, w_pm):
        m = self.m
        w_plus, w_minus = w_pm[:m], w_pm[m:]
        w = w_plus - w_minus
        z = self.x.dot(w)
        M = sp.diags((expit(z) * (1 - expit(z))).reshape((1, -1))[0])
        hess_F = (self.x.transpose().dot(M.dot(self.x)) / self.n).toarray()
        hessian = np.zeros((2 * m, 2 * m))
        hessian[:m, :m] = t * hess_F + np.diag(1 / w_plus**2 + 1 / (self.reg - w_plus - w_minus)**2)
        hessian[m:, :m] = -t * hess_F + np.diag(1 / (self.reg - w_plus - w_minus)**2)
        hessian[:m, m:] = -t * hess_F + np.diag(1 / (self.reg - w_plus - w_minus)**2)
        hessian[m:, m:] = t * hess_F + np.diag(1 / w_minus**2 + 1 / (self.reg - w_plus - w_minus)**2)
        return hessian


if __name__ == '__main__':
    x, y = load_svmlight_file('a1a.txt', zero_based=False)
    x = sp.hstack((x, np.ones((x.shape[0], 1))), format='csr')
    y[y == -1] = 0
    n, m = x.shape
    eta = 1
    t = 10
    w_pm = np.ones((2 * m, 1)) * eta / 4
    orac = OracleLogBarrier('a1a.txt', reg=1)
    print(orac.value(t, w_pm))

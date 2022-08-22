from math import sqrt
import numpy as np
import scipy.optimize as opt
import scipy as sp
import time
from collections import deque


class GoldenRatioLineSearch:
    def __init__(self):
        self.name = 'golden_ratio'
        self.step = None
        self.oracle_calls = 0

    def __call__(self, oracle, w, direction):
        f = lambda alpha: oracle.value(w + alpha * direction)
        k = (sqrt(5) - 1) / 2
        a = 0
        b = 100
        eps = 1e-8
        I = b - a
        self.oracle_calls = 0
        while I > eps:
            z = k * I
            x1 = b - z
            x2 = a + z
            fx1, fx2 = f(x1), f(x2)
            self.oracle_calls += 2
            if fx1 >= fx2:
                a = x1
            else:
                b = x2
            I = b - a
        self.step = (b + a) / 2
        return self.step


class BrentLineSearch:
    def __init__(self):
        self.name = 'brent'
        self.step = None
        self.oracle_calls = 0

    def __call__(self, oracle, w, direction):
        f = lambda alpha: oracle.value(w + alpha * direction)
        self.step, _, _, orac_calls = opt.brent(f, brack=(0, 100), full_output=True)
        self.oracle_calls = orac_calls
        return self.step


class ArmijoLineSearch:
    def __init__(self, c=0.0001):
        self.name = 'armijo'
        self.step = 1
        self.oracle_calls = 0
        self.c = c

    def __call__(self, oracle, w, direction):
        a = 1   # 2 * self.step
        cur_val, cur_grad = oracle.fuse_value_grad(w)
        self.oracle_calls = 1
        while oracle.value(w + a * direction) > cur_val + self.c * a * direction.T @ cur_grad:
            self.oracle_calls += 1
            a /= 2
        self.oracle_calls += 1
        self.step = a
        return a


class WolfeLineSearch:
    def __init__(self, c1=0.0001, c2=0.9):
        self.name = 'wolfe'
        self.step = None
        self.oracle_calls = 0
        self.c1 = c1
        self.c2 = c2

    def __call__(self, oracle, w, direction):
        self.oracle_calls = 0
        a, fc, gc = opt.line_search(oracle.value, lambda x: oracle.grad(x).reshape(-1), w, direction, c1=
                                    self.c1, c2=self.c2)[:3]
        if not a:
            method = ArmijoLineSearch()
            a = method(oracle, w, direction)
            self.oracle_calls += method.oracle_calls
        self.oracle_calls += max(fc, gc)
        return a


class NesterovLineSearch:
    def __init__(self, initial_step=1):
        self.name = 'nesterov'
        self.step = initial_step
        self.oracle_calls = 0

    def __call__(self, oracle, w, direction):
        y = w + self.step * direction
        L = 1 / self.step
        fw = oracle.value(w)
        self.oracle_calls = 1
        while oracle.value(y) > fw - direction.T.dot(y - w) \
               + L / 2 * np.linalg.norm(y - w)**2:
            L *= 2
            y = w + direction / L
            self.oracle_calls += 1
        self.oracle_calls += 1
        self.step = 2 / L
        return 1 / L


class OptimizeGD:
    def __init__(self):
        self.name = 'gradient_descend'
        self.times = []
        self.orac_calls = []
        self.n_iter = 0
        self.rel_errs = []
        self.values = []

    def __call__(self, oracle, start_point, line_search_method, tol=1e-8, maxiter=10000):
        self.times = []
        self.orac_calls = []
        self.n_iter = 0
        self.rel_errs = []
        self.values = []
        t0 = time.perf_counter()
        point = start_point
        direction = -oracle.grad(start_point)
        g0_norm = np.linalg.norm(direction) ** 2
        oracle_calls = 1
        rel_err = 1
        while self.n_iter < maxiter and (rel_err > tol or g0_norm > 1):
            alpha = line_search_method(oracle, point, direction)
            oracle_calls += line_search_method.oracle_calls
            point = point + alpha * direction
            direction = -oracle.grad(point)
            oracle_calls += 1
            grad_norm = np.linalg.norm(direction) ** 2
            if g0_norm > 1:
                g0_norm = grad_norm
            rel_err = grad_norm / g0_norm
            t = time.perf_counter() - t0
            self.times.append(t)
            self.rel_errs.append(rel_err)
            self.orac_calls.append(oracle_calls)
            self.values.append(oracle.value(point))
            self.n_iter += 1
        return point


class OptimizeNewton:
    def __init__(self):
        self.name = 'newton'
        self.times = []
        self.orac_calls = []
        self.n_iter = 0
        self.rel_errs = []
        self.values = []

    def __call__(self, oracle, start_point, line_search_method, tol=1e-8, maxiter=10000):
        self.times = []
        self.orac_calls = []
        self.n_iter = 0
        self.rel_errs = []
        self.values = []
        t0 = time.perf_counter()
        point = start_point
        g = oracle.grad(point)
        n = g.shape[0]
        g0_norm = np.linalg.norm(g) ** 2
        oracle_calls = 1
        rel_err = 1
        beta = 0.001
        id = np.eye(n)
        while self.n_iter < maxiter and (rel_err > tol or g0_norm > 1):
            h = oracle.hessian(point)
            oracle_calls += 1
            min_diag = min(np.diagonal(h))
            if min_diag > 0:
                tau = 0
            else:
                tau = -min_diag + beta
            while True:
                try:
                    c, lower = sp.linalg.cho_factor(h + tau * id)
                    break
                except np.linalg.LinAlgError:
                    tau = max(2 * tau, beta)
            direction = sp.linalg.cho_solve((c, lower), -g)
            alpha = line_search_method(oracle, point, direction)
            oracle_calls += line_search_method.oracle_calls
            point = point + alpha * direction
            g = oracle.grad(point)
            grad_norm = np.linalg.norm(g)
            oracle_calls += 1
            if g0_norm > 1:
                g0_norm = grad_norm
            rel_err = grad_norm ** 2 / g0_norm
            t = time.perf_counter() - t0
            self.n_iter += 1
            self.times.append(t)
            self.rel_errs.append(rel_err)
            self.orac_calls.append(oracle_calls)
            self.values.append(oracle.value(point))
        return point


class OptimizeHFN:
    def __init__(self):
        self.name = 'hessian_free_newton'
        self.times = []
        self.orac_calls = []
        self.n_iter = 0
        self.rel_errs = []
        self.values = []

    def __call__(self, oracle, start_point, tol_strat, tol=1e-8, maxiter=10000):
        self.times = []
        self.orac_calls = []
        self.n_iter = 0
        self.rel_errs = []
        self.values = []
        t0 = time.perf_counter()
        const_eta = 1e-4
        point = start_point
        grad = oracle.grad(point)
        n = grad.shape[0]
        n_max = 2 * n
        g0_norm = np.linalg.norm(grad)
        grad_norm = g0_norm
        oracle_calls = 1
        rel_err = 1
        while self.n_iter < maxiter and (rel_err > tol or g0_norm > 1):
            if tol_strat == 'sqrt_adaptive':
                eps = min(0.5, sqrt(grad_norm)) * grad_norm
            elif tol_strat == 'adaptive':
                eps = min(0.5, grad_norm) * grad_norm
            elif tol_strat == 'const':
                eps = const_eta * grad_norm
            else:
                raise ValueError('Такой стратегии не существует')
            zeta = np.zeros(n).reshape((-1, 1))
            r_now = grad
            d = -r_now
            j = 0
            while j < n_max:
                Hd = oracle.hessian_vec_product(point, d)
                oracle_calls += 1
                if d.T @ Hd <= 0:
                    if j == 0:
                        direction = -grad
                        break
                    else:
                        direction = zeta
                        break
                alpha = (r_now.T @ r_now) / (d.T @ Hd)
                zeta = zeta + alpha * d
                r_next = r_now + alpha * Hd
                if np.linalg.norm(r_next) < eps:
                    direction = zeta
                    break
                beta = (r_next.T @ r_next) / (r_now.T @ r_now)
                d = - r_next + beta * d
                j += 1
            else:
                direction = zeta
            method = WolfeLineSearch()
            alpha = method(oracle, point, direction)
            oracle_calls += method.oracle_calls
            point = point + alpha * direction
            grad = oracle.grad(point)
            grad_norm = np.linalg.norm(grad)
            oracle_calls += 1
            if g0_norm > 1:
                g0_norm = grad_norm
            rel_err = (np.linalg.norm(grad) / g0_norm) ** 2
            t = time.perf_counter() - t0
            self.n_iter += 1
            self.times.append(t)
            self.rel_errs.append(rel_err)
            self.orac_calls.append(oracle_calls)
            self.values.append(oracle.value(point))
        return point


class OptimizeLBFGS:
    def __init__(self, l=10):
        self.name = 'L-BFGS'
        self.times = []
        self.orac_calls = []
        self.n_iter = 0
        self.rel_errs = []
        self.values = []
        self.history = deque()
        self.l = l

    def __call__(self,  oracle, start_point, tol=1e-8, maxiter=10000):
        self.times = []
        self.orac_calls = []
        self.n_iter = 0
        self.rel_errs = []
        self.values = []
        self.history = deque()
        t0 = time.perf_counter()
        l_now = 0
        point = start_point
        grad = oracle.grad(point)
        g0_norm = np.linalg.norm(grad) ** 2
        oracle_calls = 1
        rel_err = 1
        while self.n_iter < maxiter and rel_err > tol:
            prev_point = point
            prev_grad = grad
            direction = -grad
            if l_now > 0:
                mu = [0] * l_now
                for i in range(l_now - 1, -1, -1):
                    s, y = self.history[i]  # s_k = x_{k+1} - x_k, y = grad(x_{k+1}) - grad(x_k)
                    mu[i] = (s.T @ direction) / (s.T @ y)
                    direction = direction - mu[i] * y
                s, y = self.history[0]
                direction = (s.T @ y) / (y.T @ y) * direction
                for i in range(l_now):
                    s, y = self.history[i]
                    b = (y.T @ direction) / (s.T @ y)
                    direction = direction + (mu[i] - b) * s
            method = WolfeLineSearch()
            alpha = method(oracle, point, direction)
            oracle_calls += method.oracle_calls
            point = point + alpha * direction
            s = point - prev_point
            grad = oracle.grad(point)
            y = grad - prev_grad
            self.history.append((s, y))
            if l_now == self.l:
                self.history.popleft()
            else:
                l_now += 1
            grad_norm = np.linalg.norm(grad)
            oracle_calls += 1
            if g0_norm > 1:
                g0_norm = grad_norm
            rel_err = (np.linalg.norm(grad) / g0_norm) ** 2
            t = time.perf_counter() - t0
            self.n_iter += 1
            self.times.append(t)
            self.rel_errs.append(rel_err)
            self.orac_calls.append(oracle_calls)
            self.values.append(oracle.value(point))
        return point


if __name__ == '__main__':
    from sklearn.datasets import load_svmlight_file
    from oracle import make_oracle

    orac = make_oracle('a1a.txt')

    x, y = load_svmlight_file('a1a.txt', zero_based=False)
    n = y.shape[0]
    m = x[0].shape[1] + 1
    y[y == -1] = 0
    y = y.reshape((-1, 1))

    w = np.zeros(m).reshape((-1, 1))

    optimizer = OptimizeLBFGS()
    point = optimizer(orac, w)

    print(optimizer.n_iter, optimizer.values[-1], optimizer.rel_errs[-1])
    # plt.plot(optimizer.orac_calls, optimizer.rel_errs)
    # plt.yscale('log')
    # plt.show()

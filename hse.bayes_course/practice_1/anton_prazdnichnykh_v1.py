import numpy as np
from scipy.stats import poisson, binom


def pa(params, model):
    n = params['amax'] - params['amin'] + 1
    return np.ones(n) / n, np.arange(params['amin'], params['amax'] + 1)


def pb(params, model):
    n = params['bmax'] - params['bmin'] + 1
    return np.ones(n) / n, np.arange(params['bmin'], params['bmax'] + 1)


def pa_scalar(params):
    return 1 / (params['amax'] - params['amin'] + 1)


def pb_scalar(params):
    return 1 / (params['bmax'] - params['bmin'] + 1)


def pc_ab(a, b, params, model):
    n_a = a.size
    n_b = b.size
    n_c = params['amax'] + params['bmax'] + 1
    if model == 1:
        k = np.arange(n_c).reshape(1, -1)
        pr_a = binom.pmf(k, a.reshape(-1, 1), params['p1'])
        pr_b = binom.pmf(k, b.reshape(-1, 1), params['p2'])
        return np.stack([pr_a[:, :k + 1] @ pr_b[:, :k + 1].T[::-1] for k in range(n_c)]), np.arange(n_c)
    return poisson.pmf(
        np.arange(n_c).reshape((-1, 1, 1)).repeat(n_a, axis=1).repeat(n_b, axis=2),
        a.reshape(-1, 1) * params['p1'] + b.reshape(1, -1) * params['p2']
    ), np.arange(n_c)


def pc(params, model):
    a_min, a_max = params['amin'], params['amax']
    b_min, b_max = params['bmin'], params['bmax']
    n_a = a_max - a_min + 1
    n_b = b_max - b_min + 1
    a = np.arange(a_min, a_max + 1)
    b = np.arange(b_min, b_max + 1)
    pr_c_ab, c = pc_ab(a, b, params, model)
    return pr_c_ab.sum(axis=(1, 2)) / (n_a * n_b), c


def pd_c(c, params, model):
    n_d = 2 * (params['amax'] + params['bmax']) + 1
    d = np.arange(n_d).reshape(-1, 1)
    c = c.reshape(1, -1)
    return binom.pmf(d - c.reshape(1, -1), c, params['p3'])


def pd(params, model):
    pr_c, c = pc(params, model)
    return pd_c(c, params, model).dot(pr_c), np.arange(2 * (params['amax'] + params['bmax']) + 1)


def pc_a(a, params, model):
    pr_c_ab, c = pc_ab(a, np.arange(params['bmin'], params['bmax'] + 1), params, model)
    return pr_c_ab.sum(axis=2) * pb_scalar(params), c


def pc_b(b, params, model):
    pr_c_ab, c = pc_ab(np.arange(params['amin'], params['amax'] + 1), b, params, model)
    return pr_c_ab.sum(axis=1) * pa_scalar(params), c


def pc_d(d, params, model):
    c = np.arange(params['amax'] + params['bmax'] + 1)
    pr = pd_c(c, params, model)[d].T * pc(params, model)[0].reshape(-1, 1)
    return pr / pr.sum(0), c


def pc_abd(a, b, d, params, model):
    c = np.arange(params['amax'] + params['bmax'] + 1)
    pr_d_c = pd_c(c, params, model)[d]
    pr_c_ab, _ = pc_ab(a, b, params, model)
    pr = np.transpose(pr_d_c.reshape(*pr_d_c.shape, 1, 1).repeat(a.size, 2).repeat(b.size, 3) * pr_c_ab, (1, 2, 3, 0))
    return pr / pr.sum(0), c

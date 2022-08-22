import numpy as np
from scipy.stats import poisson, binom, rv_discrete


EPS = 1e-10


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
    if model == 3:
        k = np.arange(n_c).reshape(1, -1)
        pr_a = binom.pmf(k, a.reshape(-1, 1), params['p1'])
        pr_b = binom.pmf(k, b.reshape(-1, 1), params['p2'])
        return np.stack([pr_a[:, :k + 1] @ pr_b[:, :k + 1].T[::-1] for k in range(n_c)])
    return poisson.pmf(
        np.arange(n_c).reshape((-1, 1, 1)).repeat(n_a, axis=1).repeat(n_b, axis=2),
        a.reshape(-1, 1) * params['p1'] + b.reshape(1, -1) * params['p2']
    )


def pc_b(b, params, model):
    return pc_ab(np.arange(params['amin'], params['amax'] + 1), b, params, model).sum(axis=1) * pa_scalar(params)


def pc(params, model):
    a_min, a_max = params['amin'], params['amax']
    b_min, b_max = params['bmin'], params['bmax']
    n_a = a_max - a_min + 1
    n_b = b_max - b_min + 1
    a = np.arange(a_min, a_max + 1)
    b = np.arange(b_min, b_max + 1)
    return pc_ab(a, b, params, model).sum(axis=(1, 2)) / (n_a * n_b), np.arange(params['amax'] + params['bmax'] + 1)


def pd_c(c, params, model):
    n_d = 2 * (params['amax'] + params['bmax']) + 1
    d = np.arange(n_d).reshape(-1, 1)
    c = c.reshape(1, -1)
    return binom.pmf(d - c.reshape(1, -1), c, params['p3'])


def pd(params, model):
    c = np.arange(params['amax'] + params['bmax'] + 1)
    pr_c, _ = pc(params, model)
    return pd_c(c, params, model).dot(pr_c), np.arange(2 * (params['amax'] + params['bmax']) + 1)


def pd_ab(a, b, params, model):
    n_c = params['amax'] + params['bmax'] + 1
    c = np.arange(n_c)
    return pd_c(c, params, model).dot(pc_ab(a, b, params, model).reshape(n_c, -1)).reshape(-1, a.size, b.size)


def generate(N, a, b, params, model):
    n_a = a.size
    n_b = b.size
    pr_d = pd_ab(a, b, params, model)
    cdf_d = np.cumsum(pr_d, axis=0)
    d_sampled = np.zeros((N, a.size, b.size), dtype=np.int32)
    for n_ in range(N):
        for d_, a_, b_ in zip(*np.where(np.random.rand(n_a, n_b) > cdf_d)):
            d_sampled[n_, a_, b_] = max(d_sampled[n_, a_, b_], d_)
    d_sampled += 1
    return d_sampled


def pd_b(b, params, model):
    return pd_ab(np.arange(params['amin'], params['amax'] + 1), b, params, model).sum(axis=1) * pa_scalar(params)


def pb_d(d, params, model):
    b = np.arange(params['bmin'], params['bmax'] + 1)
    pr = pd_b(b, params, model)[d].prod(axis=1).T * pb_scalar(params)
    return pr / pr.sum(axis=0), b


def pb_d_(d, params, model):
    b = np.arange(params['bmin'], params['bmax'] + 1)
    pr = pd_b(b, params, model)[d].prod(axis=1).T * pb_scalar(params)
    return pr / pd(params, model)[d].prod(axis=1), b


def pd_a(a, params, model):
    return pd_ab(a, np.arange(params['bmin'], params['bmax'] + 1), params, model).sum(axis=2) * pb_scalar(params)


def pb_ad(a, d, params, model):
    b = np.arange(params['bmin'], params['bmax'] + 1)
    pr = pd_ab(a, b, params, model)[d].prod(axis=1).T * pb_scalar(params)
    return pr / pr.sum(axis=0), b

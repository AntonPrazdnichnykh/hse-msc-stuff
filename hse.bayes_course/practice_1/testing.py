import numpy as np
from scipy.stats import poisson, binom


def pa_val(params):
    return 1 / (params["amax"] - params["amin"] + 1)


def pb_val(params):
    return 1 / (params["bmax"] - params["bmin"] + 1)


def pa(params, model):
    '''
    how many students of profile faculty there are (discrete uniform distribution)
    '''
    n = params["amax"] - params["amin"] + 1
    val = np.arange(params["amin"], params["amax"] + 1)
    return np.full(n, 1 / n), val


def pb(params, model):
    n = params["bmax"] - params["bmin"] + 1
    val = np.arange(params["bmin"], params["bmax"] + 1)
    return np.full(n, 1 / n), val


def pc_ab(a, b, params, model):
    '''
    how many students have really come on studies
    '''
    c_max = params["amax"] + params["bmax"]
    if model == 1:
        # create matrices of proba of binomial variables
        n_a = np.empty((a.size, 1))
        n_b = np.empty((b.size, 1))
        n_a[:, 0] = a
        n_b[:, 0] = b
        k = np.arange(c_max + 1)[np.newaxis, :]
        binom_proba_a = binom.pmf(k, n_a, params["p1"])
        binom_proba_b = binom.pmf(k, n_b, params["p2"])
        # create resulting tensor
        res_tensor = np.empty((c_max + 1, a.size, b.size))
        # set proba of each value in tensor
        for k in range(c_max + 1):
            flipped_binom_proba_b = np.flip(binom_proba_b[:, :k + 1].T, 0)
            res_tensor[k] = binom_proba_a[:, :k + 1].dot(flipped_binom_proba_b)
    elif model == 2:
        expectations = a[:, np.newaxis] * params["p1"] + b[np.newaxis, :] * params["p2"]
        res_tensor = poisson.pmf(k=np.arange(c_max + 1)[:, np.newaxis, np.newaxis],
                                 mu=expectations[np.newaxis, :, :])
    return np.broadcast_to(res_tensor, (c_max + 1, a.size, b.size))


def pc_a(a, params, model, distr_pc_ab=None):
    b = np.arange(params["bmin"], params["bmax"] + 1)
    c_max = params["amax"] + params["bmax"]
    if distr_pc_ab is None:
        distr_pc_ab = pc_ab(a, b, params, model)
    distr_pc_a = distr_pc_ab.sum(axis=2) * pb_val(params)
    val = np.arange(c_max + 1)
    return distr_pc_a, val


def pc_b(b, params, model, distr_pc_ab=None):
    a = np.arange(params["amin"], params["amax"] + 1)
    c_max = params["amax"] + params["bmax"]
    if distr_pc_ab is None:
        distr_pc_ab = pc_ab(a, b, params, model)
    distr_pc_b = distr_pc_ab.sum(axis=1) * pa_val(params)
    val = np.arange(c_max + 1)
    return distr_pc_b, val


def pc(params, model, distr_pc_ab=None):
    '''
    how many students have really come on studies
    '''
    a = np.arange(params["amin"], params["amax"] + 1)
    b = np.arange(params["bmin"], params["bmax"] + 1)
    c_max = params["amax"] + params["bmax"]
    if distr_pc_ab is None:
        distr_pc_ab = pc_ab(a, b, params, model)
    distr_pc = distr_pc_ab.sum(axis=(1, 2)) * pa_val(params) * pb_val(params)
    val = np.arange(c_max + 1)
    return distr_pc, val


def pd_c(d, params, model):
    c_max = params["amax"] + params["bmax"]
    k = d[:, np.newaxis] - np.arange(c_max + 1)[np.newaxis, :]
    n = np.arange(c_max + 1)[np.newaxis, :]
    return binom.pmf(k, n, params["p3"])


def pd(params, model, distr_pc_ab=None):
    c_max = params["amax"] + params["bmax"]
    d_max = 2 * c_max
    d = np.arange(d_max + 1)
    if distr_pc_ab is None:
        a = np.arange(params["amin"], params["amax"] + 1)
        b = np.arange(params["bmin"], params["bmax"] + 1)
        distr_pc_ab = pc_ab(a, b, params, model)
    distr_pd_c = pd_c(d, params, model)
    distr_pc = pc(params, model, distr_pc_ab)[0]
    distr_pd = np.dot(distr_pd_c, distr_pc)
    val = np.arange(d_max + 1)
    return distr_pd, val


def pb_a(a, params, model):
    b_cnt = params["bmax"] - params["bmin"] + 1
    res_matrix = np.empty((b_cnt, a.size))
    res_matrix.fill(pb_val(params))
    val = np.arange(params["bmin"], params["bmax"] + 1)
    return res_matrix, val


def pd_b(params, model, distr_pc_ab=None):
    b = np.arange(params["bmin"], params["bmax"] + 1)
    c_max = params["amax"] + params["bmax"]
    d_max = 2 * c_max
    d = np.arange(d_max + 1)
    if distr_pc_ab is None:
        a = np.arange(params["amin"], params["amax"] + 1)
        distr_pc_ab = pc_ab(a, b, params, model)
    distr_pd_c = pd_c(d, params, model)
    distr_pc_b = pc_b(b, params, model, distr_pc_ab)[0]
    distr_pd_b = np.dot(distr_pd_c, distr_pc_b)
    return distr_pd_b


def pb_d(d, params, model):
    a = np.arange(params["amin"], params["amax"] + 1)
    b = np.arange(params["bmin"], params["bmax"] + 1)
    distr_pc_ab = pc_ab(a, b, params, model)
    distr_pd_b = pd_b(params, model, distr_pc_ab)[d, :]
    distr_pd = pd(params, model, distr_pc_ab)[0]
    distr_pb_d = distr_pd_b.T * pb_val(params)
    distr_pb_d = distr_pb_d / distr_pb_d.sum(axis=0)
    val = np.arange(params["bmax"] - params["bmin"] + 1)
    return distr_pb_d, val


def pb_ad(a, d, params, model):
    b = np.arange(params["bmin"], params["bmax"] + 1)
    distr_pd_c = pd_c(d, params, model)
    distr_pc_ab = pc_ab(a, b, params, model)
    distr_pb_val = pb_val(params)
    distr_pc_a = distr_pc_ab.sum(axis=2) * distr_pb_val
    numerator = distr_pd_c.dot(np.swapaxes(distr_pc_ab, 0, 1)) * distr_pb_val
    denominator = distr_pd_c.dot(distr_pc_a)
    distr_b_ad = np.transpose(numerator / denominator[:, :, np.newaxis], (2, 1, 0))
    val = np.arange(params["bmax"] - params["bmin"] + 1)
    return distr_b_ad, val


def expect_of_distr(proba, val):
    return np.dot(proba, val)


def variance_of_distr(proba, val):
    return np.dot(proba, val ** 2) - np.dot(proba, val) ** 2


def arithmetic_round(x):
    if abs(x - int(x)) >= 0.5:
        return int(x) + 1
    else:
        return int(x)
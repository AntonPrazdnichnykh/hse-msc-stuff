import numpy as np


def calc_rangs(x: np.ndarray):
    rangs = np.zeros_like(x, dtype=np.float32)
    rang_cur = 0
    for item, count in zip(*np.unique(x, return_counts=True)):
        rangs[x == item] = (count * rang_cur + (count * (count + 1)) // 2) / count  # rang mean
        rang_cur = rang_cur + count
    return rangs


def montonous_conjugate(x: np.ndarray, y: np.ndarray):
    n = x.shape[0]
    assert n > 8, "Method requires size of dataset not less than 9"
    idx_sort = np.argsort(x)
    y_rangs = calc_rangs(y[idx_sort])
    p = round(n / 3)
    r1 = y_rangs[:p].sum()
    r2 = y_rangs[-p:].sum()
    diff = r2 - r1
    std_err = (n + 1 / 2) * np.sqrt(p / 6)
    conj_measure = diff / (p * (n - p))

    return diff, std_err, conj_measure



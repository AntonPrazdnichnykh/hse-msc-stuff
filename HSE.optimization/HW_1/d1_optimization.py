import numpy as np


def f(x):
    return x * x / 2, x


def optimize(f, a: float, b: float, eps: float = 1e-8):
    optimize.n_it = 0
    x, y, z = (a + b) / 2, (a + b) / 2, (a + b) / 2  # топ3 приближения минимума
    orac = f(x)
    fx, fy, fz = orac[0], orac[0], orac[0]
    dfx, dfy, dfz = orac[1], orac[1], orac[1]
    while b - a > eps:
        optimize.n_it += 1
        candidates = []
        if abs(x - y) >= eps and abs(dfx - dfy) > 0:
            u = (x * dfy - y * dfx) / (dfy - dfx)  # метод секущей поиска f'(x) = 0
            if (u > a) and (u < b):
                candidates.append(u)
        if abs(x - z) >= eps and abs(dfx - dfy) > 0:
            u = (x * dfz - z * dfx) / (dfz - dfx)
            if (u > a) and (u < b):
                candidates.append(u)
        if len(candidates):
            u = min(candidates, key=lambda v: abs(v - x))
        elif dfx > 0:  # бинпоиск нуля производной
            u = (a + x) / 2
        elif dfx < 0:
            u = (x + b) / 2
        else:
            return np.array(x)
        if (abs(a - x) < eps / 2) and (abs(x - u) < eps / 2):
            u = x + eps / 2
        elif (abs(b - x) < eps / 2) and (abs(x - u) < eps / 2):
            u = x - eps / 2
        fu, dfu = f(u)
        if fu <= fx:  # приближение u лучше, чем x
            if u > x:
                a = x
            else:
                b = x
            x, y, z = u, x, y
            fx, fy, fz = fu, fx, fy
            dfx, dfy, dfz = dfu, dfx, dfy
        else:
            if u > x:
                b = u
            else:
                a = u
            if fu <= fy or abs(y - x) < eps:  # u хуже x но лучше y
                y, z = u, y
                fy, fz = fu, fy
                dfy, dfz = dfu, dfy
            elif fu < fz or abs(z - x) < eps or abs(z - y) < eps:  # u хуже x, y, но лучше z
                z = u
                fz = fu
                dfz = dfu
    return np.array(u)

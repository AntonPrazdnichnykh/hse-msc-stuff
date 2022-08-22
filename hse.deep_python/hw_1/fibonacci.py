def fib(n: int):
    res = [0] * n
    for i in range(1, n):
        if i == 1:
            res[1] = 1
        else:
            res[i] = res[i-2] + res[i-1]
    return res

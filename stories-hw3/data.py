import numpy as np


def read_dataset(data_path: str):
    xs, ys = [], []
    with open(data_path, 'r') as f:
        for line in f:
            x, y = tuple(map(int, line.split()))
            xs.append(x)
            ys.append(y)

    return np.array(xs), np.array(ys)


if __name__ == "__main__":
    x, y = read_dataset("in.txt")
    print(x)
    print(y)

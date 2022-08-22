import numpy as np
from easy import MyMatrix
from math import floor


class HashMixin:
    def __hash__(self):
        """
        Hash function of a matrix equals to floor of sum of its elements
        """
        return floor(sum(sum(row) for row in self._data))


class HashableMatrix(HashMixin, MyMatrix):
    pass


CACHE = dict()


def matmul(x: HashableMatrix, y: HashableMatrix):
    key = hash(x), hash(y)
    if key not in CACHE:
        CACHE[key] = x @ y
    return CACHE[key]


if __name__ == "__main__":
    # Check if hash function works as expected
    l = [[1, 2], [3, 4]]
    m = HashableMatrix(l)
    assert hash(m) == 10

    # It's quite obvious that such hash function will assign same hash value for arrays which elements are some
    # permutation of one another
    A = HashableMatrix(l)
    C = HashableMatrix([[3, 4], [1, 2]])
    e = [[1, 0], [0, 1]]
    B = HashableMatrix(e)
    D = HashableMatrix(e)
    C_, D_ = np.array([[3, 4], [1, 2]]), np.array(e)
    CD_real = HashableMatrix((C_ @ D_).tolist())
    AB = matmul(A, B)
    assert (hash(A) == hash(C)) and (A != C) and (B == D) and (AB != CD_real)
    CD_cached = matmul(C, D)
    assert CD_cached == AB

    with open('artifacts/hard/A.txt', 'w') as f_A, open('artifacts/hard/B.txt', 'w') as f_B,\
            open('artifacts/hard/C.txt', 'w') as f_C, open('artifacts/hard/D.txt', 'w') as f_D, \
            open('artifacts/hard/AB.txt', 'w') as f_AB, open('artifacts/hard/CD.txt', 'w') as f_CD,\
            open('artifacts/hard/hash.txt', 'w') as f_hash:
        f_A.write(str(A))
        f_B.write(str(B))
        f_C.write(str(C))
        f_D.write(str(D))
        f_AB.write(str(AB))
        f_CD.write(str(CD_real))
        f_hash.write(str(hash(AB)))

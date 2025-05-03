import numpy as np
from copy import copy

def tridiagonalMatrixAlgorithm(A, b):
    n = len(b)

    # Вычисляем прогоночные коэффициенты
    P = np.empty((n))
    P[0] = -A[0][2] / A[0][1]
    Q = np.empty((n))
    Q[0] = b[0] / A[0][1]
    for i in range(n):
        P[i] = (-A[i][2]) / (A[i][1] + A[i][0] * P[i - 1])
        Q[i] = (b[i] - A[i][0] * Q[i - 1]) / (A[i][1] + A[i][0] * P[i - 1])

    # Обратный ход
    x = np.empty((n))
    x[n - 1] = Q[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = P[i] * x[i + 1] + Q[i]

    return x


def spline(xs, fs: list, x):
    n = len(xs)

    hs = [xs[i] - xs[i - 1] for i in range(1, n)]
    hs.insert(0, 0)

    A = np.zeros((n - 2, 3))
    A[0][0] = 2 * (hs[1] + hs[2])
    A[0][1] = hs[2]
    A[0][2] = 0
    A[n - 3][0] = 0
    A[n - 3][1] = hs[n - 2]
    A[n - 3][2] = 2 * (hs[n - 2] + hs[n - 1])
    for i in range(3, n - 1):
        A[i - 2][0] = hs[i - 1]
        A[i - 2][1] = 2 * (hs[i - 1] + hs[i])
        A[i - 2][2] = hs[i]
    
    b = np.zeros(n - 2)
    for i in range(n - 2):
        b[i] = 3 * ((fs[i + 2] - fs[i + 1]) / (hs[i + 2]) - (fs[i + 1] - fs[i]) / (hs[i + 1]))

    cs = tridiagonalMatrixAlgorithm(A, b)
    cs = np.concatenate((np.zeros(1), cs))

    as_ = copy(fs)
    as_.pop()
    as_ = np.array(as_)

    bs = np.zeros(n - 1)
    for i in range(n - 2):
        bs[i] = (fs[i + 1] - fs[i]) / hs[i + 1] - 1/3 * hs[i + 1] * (cs[i + 1] + 2 * cs[i])
    bs[n - 2] = (fs[n - 1] - fs[n - 2]) / hs[n - 1] - 2/3 * hs[n - 1] * cs[n - 2]

    ds = np.zeros(n - 1)
    for i in range(n - 2):
        ds[i] = (cs[i + 1] - cs[i]) / (3 * hs[i + 1])
    ds[n - 2] = - cs[n - 2] / (3 * hs[n - 1])

    res = 0
    for i in range(n - 1):
        if xs[i] <= x and x <= xs[i + 1]:
            res = as_[i] + bs[i] * (x - xs[i]) + cs[i] * (x - xs[i]) ** 2 + ds[i] * (x - xs[i]) ** 3

    return res

xs = [float(i) for i in input().split(" ")]
fs = [float(i) for i in input().split(" ")]
x = float(input())

y = spline(xs, fs, x)
print(f"Значение в {x}: ", y)

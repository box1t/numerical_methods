import numpy as np
from random import randint
import os 

def splitting(x0, xk, h):
    xs = []
    x = x0
    while x < xk:
        xs.append(x)
        x += h
    xs.append(xk)
    return xs

# Рунге-Кутт 4го порядка
p = 4
as_ = [0, 0.5, 0.5, 1]
bs = [[0.5], [0, 0.5], [0, 0, 1]]
cs = [1/6, 1/3, 1/3, 1/6]

def getKs(x: float, y: np.ndarray, h):
    dim = y.shape[0]
    Ks = np.empty((p, dim))

    for i in range(p):
        newX = x + as_[i] * h
        newY = np.copy(y)
        for j in range(i):
            newY += bs[i - 1][j] * Ks[j]

        K = h * f(newX, newY)
        Ks[i] = K

    return Ks

def getDeltaY(x: float, y: np.ndarray, h):
    Ks = getKs(x, y, h)
    dim = Ks.shape[1]
    sum_ = np.zeros(dim)
    for i in range(p):
        sum_ += cs[i] * Ks[i]
    return sum_

def RungeKutta(xs: list, y0: np.ndarray, h):
    N = len(xs) - 1
    dim = y0.shape[0]
    ys = np.empty((N + 1, dim))
    ys[0] = y0

    for k in range(1, N + 1):
        ys[k] = ys[k - 1] + getDeltaY(xs[k - 1], ys[k - 1], h)

    return ys

def RungeError(ys: np.ndarray, ys2: np.ndarray, p):
    k = 2
    error = 0
    for i in range(ys.shape[0]):
        error = max(error, abs(ys2[i * 2][0] - ys[i][0]) / (k ** p - 1))
    return error

def Shooting(xs, y0, h, eps):
    eta0 = randint(-2166, 2166)
    eta1 = randint(-2166, 2166)

    ys0 = RungeKutta(xs, np.array([eta0, y0]), h)
    ys1 = RungeKutta(xs, np.array([eta1, y0]), h)
    F0 = ys0[-1][0] - y1(ys0)
    F1 = ys1[-1][0] - y1(ys1)

    iter = 1
    while True:
        eta = eta1 - (eta1 - eta0) / (F1 - F0) * F1
        ys = RungeKutta(xs, np.array([eta, y0]), h)
        F0 = F1
        F1 = ys[-1][0] - y1(ys)

        if abs(F1) < eps:
            return ys, iter, eta
        
        iter += 1
        eta0 = eta1
        eta1 = eta


def f(x: float, y: np.ndarray):
    return np.array([
        y[1],
        y[1] / 2
        if x == 1 else
        (y[0] - (x - 3) * y[1]) / (x ** 2 - 1)
    ])

def getTrueY(x):
    return x - 3 + 1 / (x + 1)

def y1(ys: np.ndarray):
    return -0.75 - ys[-1][1]

a = 0
b = 1
y0 = 0
h = 0.125
eps = 1e-9

print(f"Шаг: {h}")
print(f"Точность: {eps}")

xs = splitting(a, b, h)
ysRungeKutta = RungeKutta(xs, np.array([-2, y0]), h)
ysShooting, iterShooting, eta = Shooting(xs, y0, h, eps)

print(f"Итераций в стрельбе: {iterShooting}, Вычисленная y(0) = {eta}")

for i in range(len(xs)):
    y = getTrueY(xs[i])

    print(f"xk = {np.round(xs[i], 5)}, y(xk) = {np.round(y, 5)}")

    errorRungeKutta = abs(ysRungeKutta[i][0] - y)
    errorShooting = abs(ysShooting[i][0] - y)

    print(f"\tРунге-Кутт: yk = {np.round(ysRungeKutta[i][0], 5)}, e = {np.round(errorRungeKutta, 16)}")
    print(f"\tСтрельба:   yk = {np.round(ysShooting[i][0], 5)}, e = {np.round(errorShooting, 16)}")


# Считаем для шага в два раза короче, чтобы применить оценку Рунге
h2 = h / 2

xs2 = splitting(a, b, h2)
ysShooting2, iterShooting, eta = Shooting(xs2, y0, h2, eps)
print("===================================================================")
print(f"Апостериорная оценка погрешности по Рунге: {RungeError(ysShooting, ysShooting2, 4)}")
import math
from copy import copy

def f(x):
    return math.exp(x) + x

def M(x):
    return math.exp(x)

def lagrange(xs, x):
    n = len(xs)
    res = 0

    for i in range(n):
        resCur = f(xs[i])
        for j in range(n):
            if (i == j):
                continue
            resCur *= (x - xs[j]) / (xs[i] - xs[j])
        res += resCur

    return res

def newton(xs, x):
    n = len(xs)
    res = f(xs[0])

    polynom = 1
    diffsPrev = [f(x) for x in xs]
    for i in range(2, n + 1): # сколько аргументов у разделенной разности
        diffsCur = []
        polynom *= x - xs[i - 2]
        for j in range(n - i + 1): # с какого икса начинаем
            diffsCur.append((diffsPrev[j] - diffsPrev[j + 1]) / (xs[j] - xs[j + i - 1]))
        res += polynom * diffsCur[0]
        diffsPrev = copy(diffsCur)

    return res

def error(xs, x):
    n = len(xs)
    res = M(xs[-1]) / math.factorial(n)
    for i in range(n):
        res *= x - xs[i]
    return abs(res)

xs = [float(i) for i in input().split(" ")]
x = float(input())

print("Истинное значение: ", f(x))

y = lagrange(xs, x)
print("Лагранж: ", y)

y = newton(xs, x)
print("Ньютон: ", y)

print("Погрешность: ", error(xs, x))
# 3x1 - cos(x2) = 0
# 3x2 - exp(x1) = 0

# 0.0001

import numpy as np
import matplotlib.pyplot as plt

def get_LU_decomposition(A):
    n = len(A)
    LU = np.copy(A)
    swaps = []
    for k in range(n): # обнуляемый столбец
        if (LU[k][k] == 0): # Ищем ненулевой элемент
            ind = -1
            for i in range(k + 1, n):
                if LU[i][i] != 0:
                    ind = i
                    break
            if ind == -1:
                continue
            LU[[k, ind]] = LU[[ind, k]] # Меняем местами строки если нашли строку с ненулевым элементом
            swaps.append((k, ind))

        for i in range(k + 1, n): # текущая строка
            mu = LU[i][k] / LU[k][k]
            for j in range(k, n): # текущая столбец
                if (j == k):
                    LU[i][j] = mu
                else:
                    LU[i][j] -= mu * LU[k][j]

    return (LU, swaps)

def solve_system(LU, swaps, b):
    n = len(LU)

    b = np.copy(b)

    # Меняем строки в столбце свободных членов в соответствие с заменами строк в исходной матрице
    for swap in swaps:
        b[[swap[0], swap[1]]] = b[[swap[1], swap[0]]]

    # Lz = b
    z = np.zeros(n)
    for i in range(n):
        sum_ = sum([LU[i][j] * z[j] for j in range(i)])
        z[i] = b[i] - sum_

    # Ux = z
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        sum_ = sum([LU[i][j] * x[j] for j in range(n - 1, i, -1)])
        x[i] = (z[i] - sum_) / LU[i][i]

    return x


def f(x):
    return np.array([
        3 * x[0] - np.cos(x[1]),
        3 * x[1] - np.exp(x[0])
    ])
def fDer(x):
    return np.array([
        [3, np.sin(x[1])],
        [-np.exp(x[0]), 3]
    ])
def newton(x0, eps):
    xPrev = x0
    iter = 0
    while (True):
        iter += 1
        (LU, swaps) = get_LU_decomposition(fDer(xPrev))
        xDelta = solve_system(LU, swaps, -f(xPrev))
        xCur = xPrev + xDelta
        if np.linalg.norm(xCur - xPrev, np.inf) < eps:
            break
        xPrev = xCur
    return xCur, iter


def phi(x):
    return np.array([
        np.cos(x[1]) / 3,
        np.exp(x[0]) / 3
    ])


def simpleIterations(x0, q, eps):
    xPrev = x0
    iter = 0
    while (True):
        iter += 1
        xCur = phi(xPrev)
        error = q / (1 - q) * np.linalg.norm(xCur - xPrev, np.inf)
        if error < eps:
            break
        xPrev = xCur
    return xCur, iter


q = np.e / 3
eps = float(input("Точность: "))

x0 = np.array([0, 0.3])

newtonAns, iter = newton(x0, eps)
print("Метод Ньютона")
print("\tКорень: ", newtonAns)
print("\tКоличество итераций: ", iter)

simpleIterationsAns, iter = simpleIterations(x0, q, eps)
print("Метода простых итераций")
print("\tКорень: ", simpleIterationsAns)
print("\tКоличество итераций: ", iter)


x_range = np.linspace(-0.5, 1.0, 400) 
y_range = np.linspace(-0.5, 1.0, 400)

y_vals_eq1 = y_range
x_vals_eq1 = np.cos(y_vals_eq1) / 3

x_vals_eq2 = x_range
y_vals_eq2 = np.exp(x_vals_eq2) / 3


plt.figure(figsize=(8, 6))

plt.plot(x_vals_eq1, y_vals_eq1, label='$3x_1 - \cos(x_2) = 0$', color='blue')
plt.plot(x_vals_eq2, y_vals_eq2, label='$3x_2 - e^{x_1} = 0$', color='red')

plt.scatter(newtonAns[0], newtonAns[1], color='green', zorder=5, label='Корень (Ньютон)', s=50)
if 'simpleIterationsAns' in locals():
    plt.scatter(simpleIterationsAns[0], simpleIterationsAns[1], color='purple', zorder=5, label='Корень (Простые итерации)', s=50, marker='X')


plt.title('Графики функций системы и ее корни')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.grid(True)
plt.legend()
plt.axis('equal') 
plt.axhline(0, color='grey', lw=0.5)
plt.axvline(0, color='grey', lw=0.5)
plt.savefig('graph_2_2.png') 


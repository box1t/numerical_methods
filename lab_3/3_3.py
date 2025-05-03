import numpy as np
import matplotlib.pyplot as plt

def getValue(as_: list, x):
    value = 0
    for i in range(len(as_)):
        value += as_[i] * x ** i
    return value

n = int(input("Степень многочлена: ")) + 1
xs = np.array([-3, -2, -1, 0, 1, 2])
ys = np.array([-2.9502, -1.8647, -0.63212, 1.0, 3.7183, 9.3891])
N = len(xs)

A = np.zeros((n, n))
b = np.zeros(n)
for i in range(n):
    for j in range(N):
        b[i] += ys[j] * xs[j] ** i

    for j in range(n):
        for k in range(N):
            A[i][j] += xs[k] ** (i + j)

as_ = np.linalg.solve(A, b)

print("Приближающий многочлен:")
polynom = []
for i in range(len(as_)):
    polynom.append(f"{np.round(as_[i], 4)} * x^{i}")
print(" + ".join(polynom))

fs = [getValue(as_, x) for x in xs]

error = sum([(fs[i] - ys[i]) ** 2 for i in range(N)])
print(f"Сумма квадратов ошибок: {np.round(error, 4)}")

plt.plot(xs, ys, linestyle='-', color=(1, 0, 0), label=f"Функция")
plt.plot(xs, fs, linestyle='-', color=(0, 0, 1), label=f"Приближение")
plt.legend()
plt.grid(True)
plt.savefig('3_3.png')
import numpy as np
import matplotlib.pyplot as plt

def getValue(as_: list, x):
    value = 0
    for i in range(len(as_)):
        value += as_[i] * x ** i
    return value

degree_input = int(input("Степень многочлена: "))

if degree_input > 3:
    print(f"Предупреждение: Запрошенная степень многочлена ({degree_input}) выше рекомендованной (3). Расчет будет выполнен для степени 3.")
    degree_input = 3

n = degree_input + 1

xs = np.array([-3, -2, -1, 0, 1, 2])
ys = np.array([-2.9502, -1.8647, -0.63212, 1.0, 3.7183, 9.3891])
N = len(xs)

if len(set(xs)) != len(xs):
    raise ValueError("Входные точки x должны быть уникальными.")

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

A_times_a = A @ as_

is_solution_correct = np.allclose(A_times_a, b)

print(f"\nПроверка решения системы A*a = b: {is_solution_correct}")
if not is_solution_correct:
    print("Внимание: Найденные коэффициенты могут не являться точным решением системы.")
    print("Разница (A*a - b):", A_times_a - b) 

fs = [getValue(as_, x) for x in xs]

for i in range(N):
  if not (min(xs) <= xs[i] <= max(xs)):
    raise ValueError(f"Значение x[{i}] = {xs[i]} выходит за пределы интерполяционного отрезка.")


error = sum([(fs[i] - ys[i]) ** 2 for i in range(N)])
print(f"Сумма квадратов ошибок: {np.round(error, 4)}")

plt.figure(figsize=(10, 6))
plt.plot(xs, ys, linestyle='-', color=(1, 0, 0), label=f"Функция")
plt.plot(xs, fs, linestyle='-', color=(0, 0, 1), label=f"Приближение")
plt.legend()
plt.grid(True)
plt.savefig('3_3.png')
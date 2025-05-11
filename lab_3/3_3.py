import numpy as np
import matplotlib.pyplot as plt
import os

def getValue(as_: list, x):
    """Вычисляет значение многочлена с коэффициентами as_ в точке x."""
    value = 0
    for i in range(len(as_)):
        value += as_[i] * x ** i
    return value

degree_input = int(input("Степень многочлена: "))

recommended_degree = 3
if degree_input > recommended_degree:
    print(f"Предупреждение: Запрошенная степень многочлена ({degree_input}) выше рекомендованной ({recommended_degree}). Расчет будет выполнен для степени {recommended_degree}.")
    degree_input = recommended_degree

n = degree_input + 1

xs = np.array([-3, -2, -1, 0, 1, 2])
ys = np.array([-2.9502, -1.8647, -0.63212, 1.0, 3.7183, 9.3891])
#ys = np.array([-2.9502, -1.8647, 1.63212, -2.0, -3.7183, -9.3891])
N = len(xs) 

if len(set(xs)) != N: 
    raise ValueError("Входные точки x должны быть уникальными.")

A = np.zeros((n, n))
b = np.zeros(n)

for i in range(n):
    for j in range(N):
        b[i] += ys[j] * xs[j] ** i 

    for j in range(n):
        for k in range(N):
            A[i][j] += xs[k] ** (i + j) 

try:
    as_ = np.linalg.solve(A, b)
except np.linalg.LinAlgError:
    print("Ошибка: Матрица A вырождена. Невозможно найти единственное решение. Возможно, недостаточно уникальных точек или выбрана слишком высокая степень.")
    exit() 

print("\nКоэффициенты приближающего многочлена:")
polynom_terms = []
for i in range(len(as_)):
    coeff_str = f"{np.round(as_[i], 4)}"
    if i == 0:
        polynom_terms.append(coeff_str)
    elif i == 1:
         polynom_terms.append(f"{coeff_str} * x")
    else:
        polynom_terms.append(f"{coeff_str} * x^{i}")
print(" + ".join(polynom_terms))

A_times_a = A @ as_
is_solution_correct = np.allclose(A_times_a, b)

print(f"\nПроверка выполнения нормальных уравнений (условие равенства нулю частных производных): {is_solution_correct}")
if not is_solution_correct:
    print("Внимание: Найденные коэффициенты могут не являться точным решением системы с высокой численной точностью.")

fs = [getValue(as_, x) for x in xs]

error = sum([(fs[i] - ys[i]) ** 2 for i in range(N)])
print(f"Сумма квадратов ошибок: {np.round(error, 6)}")

output_dir = '/home/snowwy/Desktop/MAI/_math/8_Численные_методы/numerical_methods/lab_3/src'
os.makedirs(output_dir, exist_ok=True)
graph_filepath = os.path.join(output_dir, '3_3.png')


plt.figure(figsize=(10, 6))

plt.plot(xs, ys, 'o', color=(1, 0, 0), label=f"Функция")

plt.plot(xs, fs, 'o-', color=(0, 0, 1), label=f"Приближение")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Аппроксимация многочленом методом наименьших квадратов")
plt.legend()
plt.grid(True)
plt.savefig(graph_filepath)
print(f"График сохранен в файл {graph_filepath}")

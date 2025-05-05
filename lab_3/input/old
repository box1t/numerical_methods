import numpy as np
from copy import copy
import matplotlib.pyplot as plt
import sys

countPoints = int(1e3)

def progonka(A, b):
    n = len(b) 
    if A.shape[0] != n or A.shape[1] != 3:
        print("Ошибка: Неверные размеры матрицы A в progonka.")
        return None

    P = np.empty((n))
    Q = np.empty((n))
    x = np.empty((n))
    if A[0][1] == 0:
         print("Ошибка: Деление на ноль в progonka при расчете P[0].")
         return None
    P[0] = -A[0][2] / A[0][1]
    Q[0] = b[0] / A[0][1]

    for i in range(n):
        denominator = A[i][1] + A[i][0] * (P[i-1] if i > 0 else 0)
        if denominator == 0:
             print(f"Ошибка: Деление на ноль в progonka на шаге прямого хода {i}.")
             return None
        P[i] = (-A[i][2]) / denominator
        Q[i] = (b[i] - A[i][0] * (Q[i-1] if i > 0 else 0)) / denominator
    x[n - 1] = Q[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = P[i] * x[i + 1] + Q[i]

    return x

def spline(xs, fs: list, x):
    n = len(xs)
    if n < 3:
        return None
    h_values = [xs[i+1] - xs[i] for i in range(n-1)] 
    if any(h <= 0 for h in h_values):
         print("Ошибка: Обнаружен неположительный размер шага в расчете сплайна.")
         return None 

    hs_orig = [xs[i] - xs[i - 1] for i in range(1, n)]
    hs_padded = copy(hs_orig)
    hs_padded.insert(0, 0) 

    num_internal_c = n - 2 

    A = np.zeros((num_internal_c, 3))
    b = np.zeros(num_internal_c)

    if num_internal_c > 0: 
        A[0][0] = 2 * (hs_padded[1] + hs_padded[2])
        A[0][1] = hs_padded[2]
        A[0][2] = 0
        if num_internal_c > 1:
            A[n - 3][0] = 0
            A[n - 3][1] = hs_padded[n - 2]
            A[n - 3][2] = 2 * (hs_padded[n - 2] + hs_padded[n - 1])
        for i in range(3, n - 1):
             if i - 2 < num_internal_c: 
                 A[i - 2][0] = hs_padded[i - 1]
                 A[i - 2][1] = 2 * (hs_padded[i - 1] + hs_padded[i])
                 A[i - 2][2] = hs_padded[i]
    for i in range(n - 2):
        b[i] = 3 * ((fs[i + 2] - fs[i + 1]) / (hs_padded[i + 2]) - (fs[i + 1] - fs[i]) / (hs_padded[i + 1]))
    cs_internal = progonka(A, b)

    if cs_internal is None:
         print("Ошибка: Не удалось вычислить коэффициенты сплайна (TDMA завершился с ошибкой).")
         return None 

    cs = np.concatenate((np.zeros(1), cs_internal))
    as_ = np.array(fs[:-1]) 
    bs = np.zeros(n - 1) 
    ds = np.zeros(n - 1) 
    for i in range(n - 2): 
        bs[i] = (fs[i + 1] - fs[i]) / h_values[i] - 1/3 * h_values[i] * (cs[i + 1] + 2 * cs[i])
        ds[i] = (cs[i + 1] - cs[i]) / (3 * h_values[i])

    bs[n - 2] = (fs[n - 1] - fs[n - 2]) / h_values[n - 2] - 2/3 * h_values[n - 2] * cs[n - 2]
    ds[n - 2] = - cs[n - 2] / (3 * h_values[n - 2])

    res = 0
    found_interval = False
    for i in range(n - 1):
        if xs[i] <= x <= xs[i + 1]:
            res = as_[i] + bs[i] * (x - xs[i]) + cs[i] * (x - xs[i]) ** 2 + ds[i] * (x - xs[i]) ** 3
            found_interval = True
            break 
    return res if found_interval else 0



print("Программа для кубической сплайн-интерполяции.")
print("Пожалуйста, введите исходные данные.")

try:
    xs_str = input("Введите значения x через пробел (строго по возрастанию): ")
    fs_str = input("Введите значения f(x) через пробел (соответствуют значениям x): ")

    xs = [float(i) for i in xs_str.split()]
    fs = [float(i) for i in fs_str.split()]

except ValueError:
    print("Ошибка ввода: Убедитесь, что введены только числа, разделенные пробелами.")
    sys.exit(1) 

if len(xs) != len(fs):
    print("Ошибка: Количество значений x и f(x) должно совпадать.")
    sys.exit(1)

n_points = len(xs)
if n_points < 2:
     print("Ошибка: Необходимо как минимум две точки для интерполяции.")
     sys.exit(1)

xs_np = np.array(xs)

if not np.all(xs_np[:-1] <= xs_np[1:]):
     print("Ошибка: Значения x должны быть отсортированы по возрастанию.")
     sys.exit(1)

if n_points > 1 and not np.all(np.diff(xs_np) > 0):
    print("Ошибка: Значения x должны быть уникальными и строго возрастать.")
    sys.exit(1)


print("\n--- Расчет значения сплайна в одной точке ---")
try:
    x_calc = float(input("Введите значение x для расчета сплайна в этой точке: "))
except ValueError:
    print("Ошибка ввода: Убедитесь, что введено число для точки расчета.")
    sys.exit(1) 

if n_points >= 2:
    if not (xs[0] <= x_calc <= xs[-1]):
        print(f"Внимание: Точка {x_calc} ({xs[0]} <= x <= {xs[-1]}) находится вне интервала интерполяции.")
        print("Значение сплайна в этой точке не может быть надежно вычислено методом интерполяции.")
    else:
        if n_points < 3:
            print(f"Недостаточно точек ({n_points}) для вычисления кубического сплайна.")
            print(f"Значение сплайна в точке {x_calc} не может быть вычислено.")
        else:
            y_calc = spline(xs, fs, x_calc)

            if y_calc is not None:
                 print(f"Значение сплайна в точке {x_calc}: {y_calc}")
            else:
                 print(f"Не удалось вычислить значение сплайна в точке {x_calc} из-за внутренней ошибки при расчете коэффициентов.")

else: 
     pass 
print("\n--- Построение графика ---")

if n_points >= 2:
    x_plot = np.linspace(xs[0], xs[-1], countPoints)

    y_plot = []
    can_plot_spline = False
    if n_points >= 3:
         y_plot_results = []
         spline_calculation_failed = False
         for curX in x_plot:
             spline_val = spline(xs, fs, curX)
             if spline_val is None:
                 spline_calculation_failed = True
                 break
             y_plot_results.append(spline_val)
         if spline_calculation_failed:
              print("Ошибка при вычислении значений для графика сплайна. График сплайна не будет построен.")
              y_plot = [0] * len(x_plot) 
              can_plot_spline = False 
         else:
              y_plot = y_plot_results
              can_plot_spline = True

    else: 
         print(f"Недостаточно точек ({n_points}) для построения кубического сплайна.")
         print("На график будут нанесены только исходные точки.")
         y_plot = [0] * len(x_plot)
         can_plot_spline = False 

    plt.figure(figsize=(10, 6))
    plt.plot(xs, fs, 'o-', color=(1, 0, 0), label="Исходные точки")

    if can_plot_spline:
         plt.plot(x_plot, y_plot, linestyle='-', color=(0, 0, 1), label="Приближение сплайном")

    plt.legend()
    plt.grid(True)
    plt.title("Кубическая сплайн-интерполяция")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.savefig('3_2.png')
    print("График сохранен в файл 3_2.png")

else: 
     pass 
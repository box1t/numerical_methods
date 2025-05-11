import math
import numpy as np
import matplotlib.pyplot as plt
import os 

def f(x):
    if isinstance(x, (int, float)):
        return math.exp(x) + x
    else:
        return np.exp(x) + x

def f_derivative(x, order):
    if order < 0:
        raise ValueError("Порядок производной должен быть неотрицательным.")
    elif order == 0:
        return f(x)
    elif order == 1:
        if isinstance(x, (int, float)):
            return math.exp(x) + 1
        else:
            return np.exp(x) + 1
    else: 
        if isinstance(x, (int, float)):
            return math.exp(x)
        else:
            return np.exp(x)

def lagrange(xs, x_eval):
    n = len(xs)
    if isinstance(x_eval, (int, float)):
        res = 0.0
    else:
        res = np.zeros_like(x_eval, dtype=float)
    for i in range(n):
        term = f(xs[i])
        for j in range(n):
            if i == j:
                continue
            denominator = xs[i] - xs[j]
            if abs(denominator) < 1e-9:
                 raise ValueError(f"Ошибка в расчетах Лагранжа: Интерполяционные узлы {xs[i]} и {xs[j]} слишком близко друг к другу.")
            if isinstance(x_eval, (int, float)):
                 term *= (x_eval - xs[j]) / denominator
            else:
                term *= (x_eval - xs[j]) / denominator
        if isinstance(x_eval, (int, float)):
            res += term
        else:
            res += term
    return res


def divided_diff(xs, ys):
    n = len(xs)
    diffs = [y for y in ys]
    for i in range(1, n):
        for j in range(n - 1, i - 1, -1):
            denominator = xs[j] - xs[j - i]
            if abs(denominator) < 1e-9:
                 raise ValueError(f"Ошибка в расчетах разделенных разностей: Узлы {xs[j]} и {xs[j-i]} слишком близко друг к другу.")
            diffs[j] = (diffs[j] - diffs[j - 1]) / denominator
    return diffs

def newton(xs, x_eval):
    n = len(xs)
    ys = [f(xi) for xi in xs]
    div_diffs = divided_diff(xs, ys)
    if isinstance(x_eval, (int, float)):
        res = div_diffs[0]
        polynom = 1.0
        for i in range(1, n):
            polynom *= (x_eval - xs[i-1])
            res += div_diffs[i] * polynom
    else:
        res = np.full_like(x_eval, div_diffs[0], dtype=float)
        polynom = np.ones_like(x_eval, dtype=float)
        for i in range(1, n):
            polynom *= (x_eval - xs[i-1])
            res += div_diffs[i] * polynom
    return res

def error_bound(xs, x):
    N = len(xs) 
    n = N - 1 

    if N == 0:
        return 0.0 

    omega_N = 1.0
    for xi in xs:
        omega_N *= (x - xi)

    interval_start = min(min(xs), x)
    interval_end = max(max(xs), x)

    try:
        points_for_max = np.linspace(interval_start, interval_end, 1000)
        nth_derivative_values = f_derivative(points_for_max, N)
        M_N = np.max(np.abs(nth_derivative_values))
    except ValueError as ve:
        print(f"Ошибка при вычислении производной для мажорантной оценки: {ve}")
        return float('inf') 
    factorial_N = math.factorial(N)

    error_bound_value = M_N / factorial_N * np.abs(omega_N)
    return error_bound_value


# --- Ввод данных пользователем ---

while True:
    try:
        xs_input = input("Введите узлы интерполяции через пробел (например: 0 1 2 3): ")
        xs = np.array([float(i) for i in xs_input.split()])

        if len(xs) < 1:
            print("Ошибка: Необходимо ввести хотя бы один узел интерполяции.")
            continue

        if len(np.unique(xs)) != len(xs):
            print("Ошибка: Узлы интерполяции должны быть различными.")
            continue

        xs = np.sort(xs)

        x_input = input(f"Введите точку для численных расчетов (не обязательно в интервале [{xs[0]}, {xs[-1]}]): ")
        x_calc = float(x_input)

        if not (xs[0] <= x_calc <= xs[-1]):
            print(f"Предупреждение: Точка {x_calc} находится вне интервала интерполяции [{xs[0]}, {xs[-1]}].")
            print("Интерполяция за пределами отрезка (экстраполяция) может давать менее точные результаты.")
        break 

    except ValueError:
        print("Ошибка ввода: Пожалуйста, вводите только числа, разделенные пробелом.")
    except Exception as e:
        print(f"Произошла непредвиденная ошибка при вводе: {e}")


print("\n--- Численные расчеты в точке {:.4f} ---".format(x_calc))
print("Истинное значение f({:.4f}): {:.10f}".format(x_calc, f(x_calc)))

try:
    y_lagrange_calc = lagrange(xs, x_calc)
    print("Лагранж в {:.4f}: {:.10f}".format(x_calc, y_lagrange_calc))
except ValueError as ve:
    print(f"Ошибка при расчете Лагранжа: {ve}")
    y_lagrange_calc = None


try:
    y_newton_calc = newton(xs, x_calc)
    print("Ньютон в {:.4f}: {:.10f}".format(x_calc, y_newton_calc))
except ValueError as ve:
    print(f"Ошибка при расчете Ньютона: {ve}")
    y_newton_calc = None

print("\n--- Проверка интерполяции в узлах ---")
ys_actual = [f(xi) for xi in xs]
ys_lagrange = [lagrange(xs, xi) for xi in xs]
ys_newton = [newton(xs, xi) for xi in xs]

for i, xi in enumerate(xs):
    print(f"Узел x[{i}] = {xi:.4f}:")
    print(f"  Истинное f(x[{i}]):    {ys_actual[i]:.10f}")
    print(f"  Лагранж в x[{i}]:   {ys_lagrange[i]:.10f}")
    print(f"  Ньютон в x[{i}]:    {ys_newton[i]:.10f}")

    if not np.isclose(ys_actual[i], ys_lagrange[i], atol=1e-9):
        print(f"  ПРЕДУПРЕЖДЕНИЕ: Лагранж не проходит точно через узел x[{i}]. Разница: {abs(ys_actual[i] - ys_lagrange[i]):.10e}")
    if not np.isclose(ys_actual[i], ys_newton[i], atol=1e-9):
        print(f"  ПРЕДУПРЕЖДЕНИЕ: Ньютон не проходит точно через узел x[{i}]. Разница: {abs(ys_actual[i] - ys_newton[i]):.10e}")

print("\n--- Оценка погрешности ---")
try:
    error_bound_calc = error_bound(xs, x_calc)
    print("Мажорантная оценка погрешности в {:.4f}: {:.10f}".format(x_calc, error_bound_calc))

    if y_lagrange_calc is not None:
        actual_error_lagrange = abs(f(x_calc) - y_lagrange_calc)
        print("Фактическая погрешность Лагранжа в {:.4f}: {:.10f}".format(x_calc, actual_error_lagrange))
        if actual_error_lagrange <= error_bound_calc * (1 + 1e-9):
             print("  Сходимость (Лагранж): Фактическая погрешность <= мажорантной оценки (с допуском) .")
             print("    Используется формула:     |R_{N-1}(x)| <= M_{N} / N! * omega_{N}(x), где M_{N} = max |f^(N)(xi)| для xi на интервале [min(xs), max(xs)]")
        else:
             print("  ПРЕДУПРЕЖДЕНИЕ (Лагранж): Фактическая погрешность превышает мажорантную оценку.")

    if y_newton_calc is not None:
        actual_error_newton = abs(f(x_calc) - y_newton_calc)
        print("Фактическая погрешность Ньютона в {:.4f}: {:.10f}".format(x_calc, actual_error_newton))
        if actual_error_newton <= error_bound_calc * (1 + 1e-9):
             print("  Сходимость (Ньютон): Фактическая погрешность <= мажорантной оценки (с допуском).")
             print("    Используется формула:     |R_{N-1}(x)| <= M_{N} / N! * omega_{N}(x), где M_{N} = max |f^(N)(xi)| для xi на интервале [min(xs), max(xs)]")
        else:
             print("  ПРЕДУПРЕЖДЕНИЕ (Ньютон): Фактическая погрешность превышает мажорантную оценку.")

except ValueError as ve:
    print(f"Ошибка при расчете мажорантной оценки: {ve}")
except Exception as e:
    print(f"Произошла ошибка при расчете мажорантной оценки: {e}")


# --- Построение графика ---

print("\n--- Построение графика ---")
output_dir = '/home/snowwy/Desktop/MAI/_math/8_Численные_методы/numerical_methods/lab_3/src'
os.makedirs(output_dir, exist_ok=True)
graph_filepath = os.path.join(output_dir, '3_1.png')

countPoints = 1000
x_plot = np.linspace(xs[0], xs[-1], countPoints)

if len(xs) < 2:
    print("Недостаточно узлов (меньше 2) для построения интерполяционного многочлена на интервале.")
    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, f(x_plot), linestyle='-', color=(1, 0, 0), label=f"Истинная функция: exp(x) + x")
    plt.scatter(xs, ys_actual, color='red', zorder=5, label='Узлы интерполяции') 
    plt.title('График истинной функции и узлов интерполяции')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    try:
        plt.savefig(graph_filepath)
        print(f"График сохранен в файл {graph_filepath}")
    except Exception as e:
        print(f"Ошибка при сохранении графика: {e}")
    plt.show()

else:
    fs_plot = f(x_plot)
    try:
        lagranges_plot = lagrange(xs, x_plot)
    except ValueError as ve:
        print(f"Ошибка при вычислении Лагранжа для графика: {ve}. График Лагранжа не будет построен.")
        lagranges_plot = None

    try:
        newtons_plot = newton(xs, x_plot)
    except ValueError as ve:
         print(f"Ошибка при вычислении Ньютона для графика: {ve}. График Ньютона не будет построен.")
         newtons_plot = None

    plt.figure(figsize=(10, 6))

    plt.plot(x_plot, fs_plot, linestyle='-', color=(1, 0, 0), label=f"Истинная функция: exp(x) + x")
    if lagranges_plot is not None:
        plt.plot(x_plot, lagranges_plot, linestyle='--', color=(0, 1, 0), label=f"Лагранж")
    if newtons_plot is not None:
        plt.plot(x_plot, newtons_plot, linestyle='-.', color=(0, 0, 1), label=f"Ньютон")

    plt.scatter(xs, ys_actual, color='red', zorder=5, label='Узлы интерполяции')

    plt.title('Интерполяция функции exp(x) + x')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)

    try:
        plt.savefig(graph_filepath)
        print(f"График сохранен в файл {graph_filepath}")
    except Exception as e:
        print(f"Ошибка при сохранении графика: {e}")
import numpy as np
from copy import copy
import matplotlib.pyplot as plt
import sys

countPoints = int(1e3)
tolerance = 1e-9

def progonka(A, b):
    n = len(b)
    if A.shape[0] != n or A.shape[1] != 3:
        print("Ошибка в progonka: Неверные размеры матрицы A.")
        return None

    P = np.empty((n))
    Q = np.empty((n))
    x = np.empty((n))

    if A[0][1] == 0:
         print("Ошибка в progonka: Деление на ноль при расчете P[0]. Главный диагональный элемент равен нулю.")
         return None
    P[0] = -A[0][2] / A[0][1]
    Q[0] = b[0] / A[0][1]

    for i in range(1, n):
        denominator = A[i][1] + A[i][0] * P[i-1]
        if abs(denominator) < tolerance:
             print(f"Ошибка в progonka: Деление на ноль на шаге прямого хода {i}. Знаменатель близок к нулю.")
             return None
        if i < n - 1:
            P[i] = -A[i][2] / denominator
        Q[i] = (b[i] - A[i][0] * Q[i-1]) / denominator

    x[n - 1] = Q[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = P[i] * x[i + 1] + Q[i]

    return x

def calculate_spline_coeffs(xs, fs: list):
    n = len(xs)
    if n < 3:
        return None

    h_values = [xs[i+1] - xs[i] for i in range(n-1)]
    if any(h <= 0 for h in h_values):
         print("Ошибка: Обнаружен неположительный размер шага в расчете сплайна. Точки x должны быть строго по возрастанию.")
         return None

    hs_orig = [xs[i] - xs[i - 1] for i in range(1, n)]

    num_internal_c = n - 2

    A = np.zeros((num_internal_c, 3))
    b = np.zeros(num_internal_c)

    if num_internal_c > 0:
        A[0][1] = 2 * (h_values[0] + h_values[1])
        A[0][2] = h_values[1]

        if num_internal_c > 1:
            A[n - 3][0] = h_values[n - 3]
            A[n - 3][1] = 2 * (h_values[n - 3] + h_values[n - 2])

        for i in range(1, num_internal_c - 1):
             A[i][0] = h_values[i-1]
             A[i][1] = 2 * (h_values[i-1] + h_values[i])
             A[i][2] = h_values[i]

    for i in range(num_internal_c):
        b[i] = 3 * ((fs[i + 2] - fs[i + 1]) / h_values[i+1] - (fs[i + 1] - fs[i]) / h_values[i])


    cs_internal = progonka(A, b)

    if cs_internal is None:
         print("Ошибка: Не удалось вычислить коэффициенты c (progonka завершилась с ошибкой).")
         return None

    cs = np.concatenate(([0.0], cs_internal, [0.0]))

    as_ = np.array(fs[:-1])
    bs = np.zeros(n - 1)
    ds = np.zeros(n - 1)

    for i in range(n - 1):
        h = h_values[i]
        bs[i] = (fs[i + 1] - fs[i]) / h - h/3 * (cs[i + 1] + 2 * cs[i])
        ds[i] = (cs[i + 1] - cs[i]) / (3 * h)


    print("\n--- Проверки коэффициентов сплайна ---")

    print("Проверка 1: Сплайн проходит через заданные точки (соответствует условиям в узлах сетки).")
    check1_errors = []
    for i in range(n - 1):
        S_at_xi = as_[i]
        if abs(S_at_xi - fs[i]) > tolerance:
            check1_errors.append(f"S({xs[i]}) = {S_at_xi:.10f}, ожидается {fs[i]:.10f} (интервал {i})")

        h = h_values[i]
        S_at_xiplus1 = as_[i] + bs[i] * h + cs[i] * h**2 + ds[i] * h**3
        if abs(S_at_xiplus1 - fs[i+1]) > tolerance:
             check1_errors.append(f"S({xs[i+1]}) = {S_at_xiplus1:.10f}, ожидается {fs[i+1]:.10f} (интервал {i})")

    if not check1_errors:
        print("Проверка 1 пройдена.")
    else:
        print("Проверка 1 не пройдена. Ошибки:")
        for error in check1_errors:
            print(f"  {error}")

    print("\nПроверка 2: Непрерывность первой и второй производной в внутренних узлах (дополнительные 2n-2 уравнения).")
    check2_errors = []

    for i in range(1, n - 1):
        h_prev = h_values[i-1]
        S_prime_from_left = bs[i-1] + 2 * cs[i-1] * h_prev + 3 * ds[i-1] * h_prev**2
        S_prime_from_right = bs[i]

        if abs(S_prime_from_left - S_prime_from_right) > tolerance:
            check2_errors.append(f"Первая производная в x = {xs[i]}: слева = {S_prime_from_left:.10f}, справа = {S_prime_from_right:.10f}")

    for i in range(1, n - 1):
        h_prev = h_values[i-1]
        S_double_prime_from_left = 2 * cs[i-1] + 6 * ds[i-1] * h_prev
        S_double_prime_from_right = 2 * cs[i]

        if abs(S_double_prime_from_left - S_double_prime_from_right) > tolerance:
            check2_errors.append(f"Вторая производная в x = {xs[i]}: слева = {S_double_prime_from_left:.10f}, справа = {S_double_prime_from_right:.10f}")

    if not check2_errors:
        print("Проверка 2 пройдена.")
    else:
        print("Проверка 2 не пройдена. Ошибки:")
        for error in check2_errors:
            print(f"  {error}")


    print("\nПроверка 3: Граничные условия натурального сплайна (недостающие 2 уравнения).")
    check3_errors = []

    S_double_prime_at_x0 = 2 * cs[0]
    if abs(S_double_prime_at_x0 - 0) > tolerance:
         check3_errors.append(f"S''({xs[0]}) = {S_double_prime_at_x0:.10f}, ожидается 0.")

    S_double_prime_at_xn_minus_1 = 2 * cs[n-1]
    if abs(S_double_prime_at_xn_minus_1 - 0) > tolerance:
         check3_errors.append(f"S''({xs[n-1]}) = {S_double_prime_at_xn_minus_1:.10f}, ожидается 0.")

    if not check3_errors:
        print("Проверка 3 пройдена.")
    else:
        print("Проверка 3 не пройдена. Ошибки:")
        for error in check3_errors:
            print(f"  {error}")

    print("--- Конец проверок ---")

    return as_, bs, cs, ds

def evaluate_spline(xs, as_, bs, cs, ds, x):
    n = len(xs)
    i = np.searchsorted(xs, x, side='right') - 1

    if i == n - 1 and abs(x - xs[n-1]) < tolerance:
         i = n - 2

    if i >= 0 and i < n - 1:
        delta_x = x - xs[i]
        res = as_[i] + bs[i] * delta_x + cs[i] * delta_x**2 + ds[i] * delta_x**3
        return res
    else:
        return 0.0


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
fs_np = np.array(fs)

if not np.all(np.diff(xs_np) > 0):
    print("Ошибка: Значения x должны быть уникальными и строго возрастать.")
    sys.exit(1)

spline_coeffs = None
if n_points >= 3:
    spline_coeffs = calculate_spline_coeffs(xs_np, fs_np)

    if spline_coeffs is not None:
        as_, bs, cs, ds = spline_coeffs
    else:
        print("Не удалось рассчитать коэффициенты сплайна.")
else:
     print(f"Недостаточно точек ({n_points}) для вычисления кубического сплайна. Коэффициенты не рассчитаны.")

print("\n--- Расчет значения сплайна в одной точке ---")
try:
    x_calc = float(input("Введите значение x для расчета сплайна в этой точке: "))
except ValueError:
    print("Ошибка ввода: Убедитесь, что введено число для точки расчета.")
    x_calc = None

if spline_coeffs is not None and x_calc is not None:
    if not (xs_np[0] - tolerance <= x_calc <= xs_np[-1] + tolerance):
        print(f"Внимание: Точка {x_calc} находится вне интервала интерполяции [{xs_np[0]}, {xs_np[-1]}].")
        print("Значение сплайна в этой точке является экстраполяцией и может быть ненадежным.")

    y_calc = evaluate_spline(xs_np, as_, bs, cs, ds, x_calc)
    print(f"Значение сплайна в точке {x_calc}: {y_calc}")
elif x_calc is not None:
    print("Расчет значения сплайна не выполнен, так как коэффициенты не были рассчитаны.")


print("\n--- Построение графика ---")

if spline_coeffs is not None:
    x_plot = np.linspace(xs_np[0], xs_np[-1], countPoints)

    y_plot = []
    evaluation_failed = False
    for curX in x_plot:
        y_val = evaluate_spline(xs_np, as_, bs, cs, ds, curX)
        y_plot.append(y_val)

    plt.figure(figsize=(10, 6))
    plt.plot(xs_np, fs_np, 'o-', color=(1, 0, 0), label="Исходные точки")

    plt.plot(x_plot, y_plot, linestyle='-', color=(0, 0, 1), label="Приближение сплайном")

    plt.legend()
    plt.grid(True)
    plt.title("Кубическая сплайн-интерполяция")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.savefig('3_2.png')
    print("График сохранен в файл 3_2.png")

elif n_points >= 2:
    print(f"Недостаточно точек ({n_points}) для построения кубического сплайна.")
    print("На график будут нанесены только исходные точки.")
    plt.figure(figsize=(10, 6))
    plt.plot(xs_np, fs_np, 'o-', color=(1, 0, 0), label="Исходные точки")
    plt.legend()
    plt.grid(True)
    plt.title("Исходные точки")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.savefig('3_2.png')
    print("График сохранен в файл 3_2.png")

else:
     pass
import numpy as np

def check_constant_step(xs, tolerance=1e-9):
    if len(xs) < 2:
        return True
    steps = np.diff(xs)
    if len(steps) < 2:
        return True
    first_step = steps[0]
    if not np.allclose(steps, first_step, atol=tolerance):
        return False
    return True

def getIndex(x, xs):
    if not isinstance(xs, (list, tuple, np.ndarray)):
        raise ValueError("xs должен быть списком, кортежем или массивом numpy.")
    n = len(xs)
    if n < 2:
        raise ValueError("xs должен содержать как минимум 2 точки для определения интервала.")
    if any(xs[i] >= xs[i+1] for i in range(n - 1)):
         if any(xs[i] == xs[i+1] for i in range(n - 1)):
             raise ValueError("xs содержит одинаковые или неупорядоченные точки. Точки должны быть строго возрастающими.")
         else:
             raise ValueError("xs должен быть строго отсортирован по возрастанию.")

    if x < xs[0] or x > xs[n - 1]:
        raise ValueError(f"Точка x ({x}) находится вне диапазона данных [{xs[0]}, {xs[n-1]}].")

    if isinstance(xs, np.ndarray):
        j = np.searchsorted(xs, x, side='right') - 1
        if x == xs[j] and j > 0:
             return j - 1
        if j < n - 1:
             return j
        if x == xs[n-1]:
            return n - 2
    else:
        for i in range(n - 1):
            if xs[i] <= x <= xs[i + 1]:
                 return i

        if x == xs[n-1]:
            return n - 2

    raise RuntimeError(f"Не удалось найти интервал для x={x}. Проверьте данные xs.")

def firstDerivative(x, xs, ys):
    if len(xs) != len(ys):
        raise ValueError("Длины списков xs и ys должны совпадать.")

    is_constant_step = check_constant_step(xs)
    if not is_constant_step:
        print("Внимание: Точки xs не имеют постоянного шага. "
              "Порядок точности формул может отличаться от ожидаемого O(h^2).")

    i = getIndex(x, xs)

    if i + 2 >= len(xs):
        raise ValueError(f"Недостаточно точек для вычисления производной в x={x} "
                         f"с использованием данной формулы (требуются xs[i], xs[i+1], xs[i+2]). "
                         f"Данная формула применима только для x в диапазоне [{xs[0]}, {xs[len(xs)-2]}].")

    h1 = xs[i+1] - xs[i]
    h2 = xs[i+2] - xs[i+1]
    denom_total = xs[i+2] - xs[i]

    if h1 == 0 or h2 == 0 or denom_total == 0:
         raise ZeroDivisionError(f"Обнаружены одинаковые точки в xs[{i}], xs[{i+1}] или xs[{i+2}], "
                                 f"приводящие к делению на ноль.")

    div_diff1 = (ys[i+1] - ys[i]) / h1
    div_diff2 = (ys[i+2] - ys[i+1]) / h2
    total_div_diff = (div_diff2 - div_diff1) / denom_total

    derivative = div_diff1 + total_div_diff * (2 * x - xs[i] - xs[i+1])

    return derivative

def secondDerivative(x, xs, ys):
    if len(xs) != len(ys):
        raise ValueError("Длины списков xs и ys должны совпадать.")

    is_constant_step = check_constant_step(xs)
    if not is_constant_step:
        print("Внимание: Точки xs не имеют постоянного шага. "
              "Порядок точности формул может отличаться от ожидаемого O(h^2).")

    i = getIndex(x, xs)

    if i + 2 >= len(xs):
        raise ValueError(f"Недостаточно точек для вычисления второй производной в x={x} "
                         f"с использованием данной формулы (требуются xs[i], xs[i+1], xs[i+2]). "
                         f"Данная формула применима только для x в диапазоне [{xs[0]}, {xs[len(xs)-2]}].")

    h1 = xs[i+1] - xs[i]
    h2 = xs[i+2] - xs[i+1]
    denom_total = xs[i+2] - xs[i]

    if h1 == 0 or h2 == 0 or denom_total == 0:
         raise ZeroDivisionError(f"Обнаружены одинаковые точки в xs[{i}], xs[{i+1}] или xs[{i+2}], "
                                 f"приводящие к делению на ноль.")

    div_diff1 = (ys[i+1] - ys[i]) / h1
    div_diff2 = (ys[i+2] - ys[i+1]) / h2
    total_div_diff = (div_diff2 - div_diff1) / denom_total

    derivative = 2 * total_div_diff

    return derivative

## ---- ПРИМЕРЫ ---- 

x = 0.2
xs = np.array([-0.2, 0.0, 0.2, 0.4, 0.6])
ys = np.array([-0.40136, 0.0, 0.40136, 0.81152, 1.2435])

print(f"Набор данных xs: {xs}")
print(f"Набор данных ys: {ys}")

if check_constant_step(xs):
    print("\nПроверка шага: Точки xs имеют постоянный шаг.")
else:
    print("\nПроверка шага: Внимание, точки xs не имеют постоянного шага!") 

try:
    print(f"\nВычисление производных в точке x = {x}")
    print(f"Первая производная в точке {x}: {firstDerivative(x, xs, ys)}")
    print(f"Вторая производная в точке {x}: {secondDerivative(x, xs, ys)}")
except ValueError as e:
    print(f"Ошибка при вычислении: {e}")
except ZeroDivisionError as e:
    print(f"Ошибка при вычислении (деление на ноль): {e}")
except Exception as e:
    print(f"Неожиданная ошибка: {e}")

print("\n" + "=" * 40 + "\n")

x_boundary_explanation = 0.4
n_explanation = len(xs)
max_i_explanation = n_explanation - 3
applicable_range_end = xs[n_explanation - 2]

print(f"Точка x = {x_boundary_explanation}")
print(f"Всего точек в xs: n = {n_explanation}")
print(f"Формулы используют точки xs[i], xs[i+1], xs[i+2].")
print(f"Для этого необходимо, чтобы i+2 < n, т.е. i <= n-3.")
print(f"Максимальный допустимый индекс i для данной формулы: n-3 = {max_i_explanation}")
print(f"Последний интервал [xs[i], xs[i+1]] с i <= {max_i_explanation} это [xs[{max_i_explanation}], xs[{max_i_explanation+1}]] = [{xs[max_i_explanation]}, {xs[max_i_explanation+1]}]")
print(f"Таким образом, данная формула применима для x в диапазоне [{xs[0]}, {applicable_range_end}].")
print(f"Точка x = {x_boundary_explanation} ({applicable_range_end}) является правой границей этого диапазона применимости.")
try:
    i_boundary = getIndex(x_boundary_explanation, xs)
    points_used = xs[i_boundary : i_boundary + 3]
    print(f"Для x = {x_boundary_explanation}, getIndex вернул i = {i_boundary}.")
    print(f"Используются точки xs[{i_boundary}], xs[{i_boundary+1}], xs[{i_boundary+2}], т.е. {points_used}.")

    print(f"\nВычисление производных в граничной точке x = {x_boundary_explanation}")
    print(f"Первая производная в точке {x_boundary_explanation}: {firstDerivative(x_boundary_explanation, xs, ys)}")
    print(f"Вторая производная в точке {x_boundary_explanation}: {secondDerivative(x_boundary_explanation, xs, ys)}")
except ValueError as e:
    print(f"Ошибка при вычислении: {e}")
except ZeroDivisionError as e:
    print(f"Ошибка при вычислении (деление на ноль): {e}")

print("\n" + "=" * 40 + "\n")

# Объяснение про порядок точности
print(f"Порядок точности O(h^2) в центральной точке для первой и второй производной")
print(f"Порядок точности O(h) в точках отрезка и на границах для первой и второй производной")

print("\n" + "=" * 40 + "\n")

x_out_of_formula_range = 0.5
print(f"Пример точки x = {x_out_of_formula_range} вне диапазона применимости формулы [{xs[0]}, {applicable_range_end}]:")
try:
    print(f"Первая производная в точке {x_out_of_formula_range}: {firstDerivative(x_out_of_formula_range, xs, ys)}")
    print(f"Вторая производная в точке {x_out_of_formula_range}: {secondDerivative(x_out_of_formula_range, xs, ys)}")
except ValueError as e:
    print(f"Ошибка при вычислении: {e}")
except ZeroDivisionError as e:
    print(f"Ошибка при вычислении (деление на ноль): {e}")

print("\n" + "=" * 40 + "\n")

x_out_of_data_range = 1.0
print(f"Пример точки x = {x_out_of_data_range} вне диапазона данных [{xs[0]}, {xs[-1]}]:")
try:
    print(f"Первая производная в точке {x_out_of_data_range}: {firstDerivative(x_out_of_data_range, xs, ys)}")
    print(f"Вторая производная в точке {x_out_of_data_range}: {secondDerivative(x_out_of_data_range, xs, ys)}")
except ValueError as e:
    print(f"Ошибка при вычислении: {e}")
except ZeroDivisionError as e:
    print(f"Ошибка при вычислении (деление на ноль): {e}")

print("\n" + "=" * 40 + "\n")

print(f"Пример с недостаточным количеством точек для формулы:")
xs_short = [0.0, 0.1]
ys_short = [1.0, 1.1]
x_short = 0.05
try:
     print(f"Первая производная в точке {x_short} с данными xs={xs_short}, ys={ys_short}:")
     print(f"{firstDerivative(x_short, xs_short, ys_short)}")
except ValueError as e:
     print(f"Ошибка при вычислении: {e}")

print("\n" + "=" * 40 + "\n")

print(f"Пример с неравномерным шагом (выдаст предупреждение):")
xs_uneven = [0.0, 0.1, 0.3, 0.4]
ys_uneven = [0.0, 0.1, 0.9, 1.6]
x_uneven = 0.15
try:
    print(f"Первая производная в точке {x_uneven} с данными xs={xs_uneven}, ys={ys_uneven}:")
    print(f"{firstDerivative(x_uneven, xs_uneven, ys_uneven)}")
except ValueError as e:
    print(f"Ошибка при вычислении: {e}")
except ZeroDivisionError as e:
    print(f"Ошибка при вычислении (деление на ноль): {e}") 
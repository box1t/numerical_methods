import numpy as np

def check_constant_step(xs, tolerance=1e-9):
    if len(xs) < 2:
        return True
    steps = np.diff(xs)
    if len(steps) < 2:
        return True
    first_step = steps[0]
    if abs(first_step) < tolerance and np.all(np.abs(steps) < tolerance):
         return True
    if not np.allclose(steps, first_step, atol=tolerance):
        return False
    return True

def getIndex(x, xs, tolerance=1e-9):
    if not isinstance(xs, (list, tuple, np.ndarray)):
        raise ValueError("xs должен быть списком, кортежем или массивом numpy.")
    n = len(xs)
    if n < 2:
        raise ValueError("xs должен содержать как минимум 2 точки для определения интервала.")
    if any(xs[i+1] - xs[i] < -tolerance for i in range(n - 1)):
         raise ValueError("xs должен быть строго отсортирован по возрастанию.")
    if any(abs(xs[i+1] - xs[i]) < tolerance for i in range(n - 1)):
         raise ValueError("xs содержит почти одинаковые точки. Точки должны быть строго возрастающими.")

    if x < xs[0] - tolerance or x > xs[n - 1] + tolerance:
         raise ValueError(f"Точка x ({x}) находится вне диапазона данных [{xs[0]}, {xs[n-1]}].")

    if isinstance(xs, np.ndarray):
        j = np.searchsorted(xs, x, side='right') - 1

        if j > 0 and np.isclose(x, xs[j], atol=tolerance):
             i = j - 1
        else:
             i = j

        i = np.clip(i, 0, n - 2)

        return i

    else:
        for i in range(n - 1):
            if (xs[i] < x + tolerance and xs[i+1] > x - tolerance) or \
               np.isclose(x, xs[i], atol=tolerance) or \
               np.isclose(x, xs[i+1], atol=tolerance):
                 return i

        if np.isclose(x, xs[n-1], atol=tolerance):
             return n - 2

    raise RuntimeError(f"Не удалось найти интервал для x={x}. Проверьте данные xs.")


def firstDerivative(x, xs, ys, tolerance=1e-9):
    if len(xs) != len(ys):
        raise ValueError("Длины списков xs и ys должны совпадать.")

    n = len(xs)
    if n < 3:
         raise ValueError(f"Недостаточно точек для вычисления производной в x={x} "
                          f"с использованием данной формулы (требуется минимум 3 точки). "
                          f"Получено {n} точек.")

    is_constant_step = check_constant_step(xs, tolerance)
    # Новая проверка: если шаг не постоянный, выбрасываем исключение
    if not is_constant_step:
        raise ValueError("Вычисление первой производной для неравномерной сетки не предусмотрено в данной реализации.")


    try:
        i = getIndex(x, xs, tolerance)
    except ValueError as e:
         raise ValueError(f"Ошибка при определении интервала для x={x}: {e}") from e

    if i + 2 >= n:
        applicable_range_end = xs[n-2]
        raise ValueError(f"Недостаточно точек справа от интервала [{xs[i]}, {xs[i+1]}] "
                         f"для вычисления производной в x={x} "
                         f"с использованием данной формулы (требуются xs[{i}], xs[{i+1}], xs[{i+2}]). "
                         f"Данная формула применима только для x в диапазоне [{xs[0]}, {applicable_range_end}].")

    is_central_point = np.isclose(x, xs[i+1], atol=tolerance)

    if is_central_point:
        if is_constant_step: # Эта ветка теперь выполнится только если is_constant_step == True
             print(f"  -> Точка вычисления x={x} является центральной точкой ({xs[i+1]}) используемого шаблона xs[от {i} до {i+2}] с постоянным шагом. Ожидаемый порядок точности O(h²).")
        # Ветка else (не центральная точка) теперь тоже выполнится только если шаг постоянный
        else: # Этот else технически не нужен из-за предыдущей проверки, но оставлен для ясности логики сообщений
             print(f"  -> Точка вычисления x={x} является центральной точкой ({xs[i+1]}) используемого шаблона xs[от {i} до {i+2}] с непостоянным шагом.")
    else:
         # Эта ветка тоже выполнится только если шаг постоянный
         print(f"  -> Точка вычисления x={x} находится в интервале [{xs[i]}, {xs[i+1]}] (или равна {xs[i]}) и не является центральной точкой ({xs[i+1]}) используемого шаблона xs[от {i} до {i+2}]. Ожидаемый порядок точности O(h) (для равномерной сетки).")


    h1 = xs[i+1] - xs[i]
    h2 = xs[i+2] - xs[i+1]
    denom_total = xs[i+2] - xs[i]

    if abs(h1) < tolerance or abs(h2) < tolerance or abs(denom_total) < tolerance:
         raise ZeroDivisionError(f"Обнаружены почти одинаковые точки в xs[{i}], xs[{i+1}] или xs[{i+2}], "
                                 f"приводящие к делению на ноль (шаги h1={h1}, h2={h2}, denom_total={denom_total}). "
                                 f"Проверьте данные xs в диапазоне индексов [{i}, {i+2}].")

    div_diff1 = (ys[i+1] - ys[i]) / h1
    div_diff2 = (ys[i+2] - ys[i+1]) / h2
    total_div_diff = (div_diff2 - div_diff1) / denom_total

    derivative = div_diff1 + total_div_diff * (2 * x - xs[i] - xs[i+1])

    return derivative

def secondDerivative(x, xs, ys, tolerance=1e-9):
    if len(xs) != len(ys):
        raise ValueError("Длины списков xs и ys должны совпадать.")

    n = len(xs)
    if n < 3:
         raise ValueError(f"Недостаточно точек для вычисления второй производной в x={x} "
                          f"с использованием данной формулы (требуется минимум 3 точки). "
                          f"Получено {n} точек.")

    is_constant_step = check_constant_step(xs, tolerance)
    # Новая проверка: если шаг не постоянный, выбрасываем исключение
    if not is_constant_step:
         raise ValueError("Вычисление второй производной для неравномерной сетки не предусмотрено в данной реализации.")


    try:
        i = getIndex(x, xs, tolerance)
    except ValueError as e:
         raise ValueError(f"Ошибка при определении интервала для x={x}: {e}") from e

    if i + 2 >= n:
        applicable_range_end = xs[n-2]
        raise ValueError(f"Недостаточно точек справа от интервала [{xs[i]}, {xs[i+1]}] "
                         f"для вычисления второй производной в x={x} "
                         f"с использованием данной формулы (требуются xs[{i}], xs[{i+1}], xs[{i+2}]). "
                         f"Данная формула применима только для x в диапазоне [{xs[0]}, {applicable_range_end}].")

    h1 = xs[i+1] - xs[i]
    h2 = xs[i+2] - xs[i+1]
    denom_total = xs[i+2] - xs[i]

    if abs(h1) < tolerance or abs(h2) < tolerance or abs(denom_total) < tolerance:
         raise ZeroDivisionError(f"Обнаружены почти одинаковые точки в xs[{i}], xs[{i+1}] или xs[{i+2}], "
                                 f"приводящие к делению на ноль (шаги h1={h1}, h2={h2}, denom_total={denom_total}). "
                                 f"Проверьте данные xs в диапазоне индексов [{i}, {i+2}].")

    div_diff1 = (ys[i+1] - ys[i]) / h1
    div_diff2 = (ys[i+2] - ys[i+1]) / h2
    total_div_diff = (div_diff2 - div_diff1) / denom_total

    derivative = 2 * total_div_diff

    return derivative

def run_derivative_example(description, x, xs, ys, tolerance=1e-9):
    print("=" * 40)
    print(description)
    print(f"Набор данных xs: {xs}")
    print(f"Длина массива xs: {len(xs)}")
    print(f"Набор данных ys: {ys}")

    if len(xs) > 1:
        if check_constant_step(xs, tolerance):
            print("Проверка шага: Точки xs имеют постоянный шаг.")
        else:
            print("Проверка шага: Внимание, точки xs не имеют постоянного шага! Вычисление не будет выполнено.")
    else:
        print("Проверка шага: Неприменимо (менее 2 точек).")

    # Вызываем функции производных только если точек достаточно для потенциального расчета (хотя бы 3)
    if len(xs) >= 3 and check_constant_step(xs, tolerance): # Дополнительная проверка здесь, чтобы не вызывать функции, если шаг неравномерный
         try:
             print(f"\nВычисление производных в точке x = {x}")
             first_deriv = firstDerivative(x, xs, ys, tolerance)
             print(f"Первая производная в точке {x}: {first_deriv}")

             second_deriv = secondDerivative(x, xs, ys, tolerance)
             print(f"Вторая производная в точке {x}: {second_deriv}")

         except ValueError as e:
             print(f"Ошибка при вычислении: {e}")
         except ZeroDivisionError as e:
             print(f"Ошибка при вычислении (деление на ноль): {e}")
         except Exception as e:
             print(f"Неожиданная ошибка: {e}")
    elif len(xs) < 3:
         # Сообщение о недостаточном количестве точек уже есть в функциях derivative,
         # но этот else elif блок для ясности потока в run_derivative_example.
         # В случае N=2 или N=0,1, вызов firstDerivative/secondDerivative выбросит ValueError.
         # В случае N=3+, но неравномерной сетки, мы уже выбросили ValueError выше.
         # Этот блок по сути ловит случаи, когда len(xs) < 3.
         try:
             # Просто попытка вызвать, чтобы получить стандартное сообщение об ошибке
             firstDerivative(x, xs, ys, tolerance)
         except ValueError as e:
             print(f"\nОшибка при вычислении: {e}")
         except Exception as e:
              print(f"\nНеожиданная ошибка при вычислении: {e}")


    print("=" * 40 + "\n")

run_derivative_example(
    "Пример 1: N=5, Равномерная сетка, x совпадает с xs[2] (центральная точка шаблона xs[1..3]). Ожидается O(h²).",
    x = 0.2,
    xs = np.array([-0.2, 0.0, 0.2, 0.4, 0.6]),
    ys = np.array([-0.40136, 0.0, 0.40136, 0.81152, 1.2435])
)

run_derivative_example(
    "Пример 2: N=5, Равномерная сетка, x совпадает с xs[1] (центральная точка шаблона xs[0..2]). Ожидается O(h²).",
    x = 0.0,
    xs = np.array([-0.2, 0.0, 0.2, 0.4, 0.6]),
    ys = np.array([-0.40136, 0.0, 0.40136, 0.81152, 1.2435])
)

run_derivative_example(
    "Пример 3: N=5, Равномерная сетка, x совпадает с xs[n-2] (xs[3]) и является центральной точкой шаблона xs[2..4]. Ожидается O(h²).",
    x = 0.4,
    xs = np.array([-0.2, 0.0, 0.2, 0.4, 0.6]),
    ys = np.array([-0.40136, 0.0, 0.40136, 0.81136, 1.2435])
)

run_derivative_example(
    "Пример 4: N=5, Точка x совпадает с xs[n-1], вне диапазона применимости формулы [xs[0], xs[n-2]]. Ожидается ошибка.",
    x = 0.6,
    xs = np.array([-0.2, 0.0, 0.2, 0.4, 0.6]),
    ys = np.array([-0.40136, 0.0, 0.40136, 0.81152, 1.2435])
)

run_derivative_example(
    "Пример 5: N=5, Точка x вне диапазона данных [xs[0], xs[n-1]]. Ожидается ошибка.",
    x = 1.0,
    xs = np.array([-0.2, 0.0, 0.2, 0.4, 0.6]),
    ys = np.array([-0.40136, 0.0, 0.40136, 0.81152, 1.2435])
)

run_derivative_example(
    "Пример 6: N=2. Недостаточно точек для формулы (требуется >= 3). Ожидается ошибка.",
    x = 0.05,
    xs = np.array([0.0, 0.1]),
    ys = np.array([1.0, 1.1])
)

run_derivative_example(
    "Пример 7: N=3, Равномерная сетка, x совпадает с xs[1] (центральная точка шаблона xs[0..2]). Ожидается O(h²).",
    x = 0.1,
    xs = np.array([0.0, 0.1, 0.2]),
    ys = np.array([0.0, 0.01, 0.04])
)

run_derivative_example(
    "Пример 8: N=3, Равномерная сетка, x в интервале (xs[0], xs[1]), не является центральной точкой шаблона xs[0..2]. Ожидается O(h).",
    x = 0.05,
    xs = np.array([0.0, 0.1, 0.2]),
    ys = np.array([0.0, 0.01, 0.04])
)

run_derivative_example(
    "Пример 9: N=3, x совпадает с xs[n-2] (xs[1]) и является центральной точкой шаблона xs[0..2]). Ожидается O(h²).",
    x = 0.1,
    xs = np.array([0.0, 0.1, 0.2]),
    ys = np.array([0.0, 0.01, 0.04])
)

run_derivative_example(
    "Пример 10: N=4, Неравномерный шаг. Ожидается сообщение о невозможности вычисления.",
    x = 0.15,
    xs = np.array([0.0, 0.1, 0.3, 0.4]),
    ys = np.array([0.0, 0.1, 0.9, 1.6])
)

run_derivative_example(
    "Пример 11: N=4, Неравномерный шаг. Ожидается сообщение о невозможности вычисления.",
    x = 0.3,
    xs = np.array([0.0, 0.1, 0.3, 0.4]),
    ys = np.array([0.0, 0.1, 0.9, 1.6])
)


print("=" * 40)
print("ОБЪЯСНЕНИЕ:")
print("  - Данная реализация трехточечной формулы производной предназначена для использования ТОЛЬКО на равномерной сетке.")
print("  - При входных данных с неравномерным шагом вычисления не выполняются.")
print("  - На равномерной сетке:")
print("    - В точке, совпадающей с центральным узлом (xs[i+1]) используемого шаблона xs[i]..xs[i+2]:")
print("      Ожидаемый порядок точности O(h²).")
print("    - В остальных точках интервала [xs[i], xs[i+1]] (и на границах интервалов, кроме центральной точки шаблона):")
print("      Ожидаемый порядок точности O(h).")
print("  - Формула требует минимум 3 точки и применима только для x в диапазоне [xs[0], xs[n-2]].")
print("=" * 40 + "\n")
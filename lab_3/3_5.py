import math
import sys  

def f(x):
    denominator = 256 - x ** 4
    if abs(denominator) < sys.float_info.epsilon:
        raise ValueError(f"Знаменатель функции f(x) равен нулю при x = {x}")
    return 1 / denominator

def splitting(x0, xk, h):
    if h <= 0:
        raise ValueError("Шаг h должен быть положительным числом.")
    if x0 >= xk:
        raise ValueError("Начальная точка x0 должна быть меньше конечной точки xk.")
    xs = []
    x = x0
    while x < xk + sys.float_info.epsilon * 10:
        if x > xk: 
             x = xk
        xs.append(x)
        x += h
    if abs(xs[-1] - xk) > sys.float_info.epsilon * 10:
         xs.append(xk)
    if len(xs) > 1 and abs(xs[-1] - xs[-2]) < sys.float_info.epsilon * 10 and abs(xs[-1] - xk) < sys.float_info.epsilon * 10:
         xs.pop(-2)
    return xs

def rectangles(x0, xk, h):
    xs = splitting(x0, xk, h)
    if len(xs) < 2:
        return 0.0 if x0 == xk else f((x0 + xk) / 2) * (xk - x0) 
    integral = 0
    for i in range(len(xs) - 1):
        midpoint = (xs[i] + xs[i+1]) / 2
        if midpoint < x0 or midpoint > xk:
             print(f"Предупреждение: Точка {midpoint} вышла за границы отрезка интегрирования [{x0}, {xk}]", file=sys.stderr)
        try:
            integral += h * f(midpoint)
        except ValueError as e:
            print(f"Ошибка при вычислении f({midpoint}): {e}", file=sys.stderr)
            raise 
    return integral

def trapezoids(x0, xk, h):
    xs = splitting(x0, xk, h)
    if len(xs) < 2:
        return 0.0 if x0 == xk else 0.5 * (f(x0) + f(xk)) * (xk - x0) 
    integral = 0
    for i in range(len(xs) - 1):
        x_i = xs[i]
        x_i_plus_1 = xs[i+1]
        if x_i < x0 or x_i > xk or x_i_plus_1 < x0 or x_i_plus_1 > xk:
             print(f"Предупреждение: Граничные точки {x_i}, {x_i_plus_1} вышли за границы отрезка интегрирования [{x0}, {xk}]", file=sys.stderr)
        try:
            integral += 0.5 * h * (f(x_i) + f(x_i_plus_1))
        except ValueError as e:
            print(f"Ошибка при вычислении f({x_i}) или f({x_i_plus_1}): {e}", file=sys.stderr)
            raise 
    return integral

def simpson(x0, xk, h):
    xs = splitting(x0, xk, h)
    n = len(xs) - 1 
    if n <= 0:
         return 0.0 
    if n % 2 != 0:
        raise ValueError(f"Для метода Симпсона требуется четное число интервалов. Получено {n}.")
    integral = 0
    for i in range(0, n, 2):
        x_i = xs[i]
        x_i_plus_1 = xs[i+1]
        x_i_plus_2 = xs[i+2]
        midpoint = (x_i + x_i_plus_2) / 2
        if x_i < x0 or x_i > xk or x_i_plus_1 < x0 or x_i_plus_1 > xk or x_i_plus_2 < x0 or x_i_plus_2 > xk:
             print(f"Предупреждение: Точки {x_i}, {x_i_plus_1}, {x_i_plus_2} за границами отрезка интегрирования [{x0}, {xk}]", file=sys.stderr)
        try:
            integral += h/3 * (f(x_i) + 4 * f(x_i_plus_1) + f(x_i_plus_2))
        except ValueError as e:
            print(f"Ошибка при вычислении f({x_i}), f({x_i_plus_1}) или f({x_i_plus_2}): {e}", file=sys.stderr)
            raise 
    return integral

def rungeError(values, hs, p):
    if len(values) < 2 or len(hs) < 2:
        raise ValueError("Для метода Рунге требуется как минимум два значения интеграла и два шага.")
    if hs[0] <= 0 or hs[1] <= 0:
         raise ValueError("Шаги h должны быть положительными.")
    if p <= 0:
         raise ValueError("Порядок точности p должен быть положительным.")
    k = hs[0] / hs[1]
    denominator = k ** p - 1
    if abs(denominator) < sys.float_info.epsilon:
        raise ValueError(f"Знаменатель в методе Рунге близок к нулю (k^p - 1), k={k}, p={p}.")
    return (values[1] - values[0]) / denominator

def runge(values, hs, p):
    if len(values) < 2 or len(hs) < 2:
        raise ValueError("Для метода Рунге требуется как минимум два значения интеграла и два шага.")
    return values[1] + rungeError(values, hs, p)

orderRectangles = 2
orderTrapezoids = 2
orderSimpson = 4

x0 = -2
xk = 2
hs = [1, 0.5]

try:
    integral = (math.log(6) - math.log(2) + 2 * math.atan(0.5)) / 128 
except Exception as e:
    print(f"Ошибка при вычислении истинного значения интеграла: {e}", file=sys.stderr)
    integral = None 

if integral is not None:
    print(f"Истинное значение: {integral}")

    integralsRectangles = []
    errorsRectangles = []
    integralsTrapezoids = []
    errorsTrapezoids = []
    integralsSimpson = []
    errorsSimpson = []

    for h in hs:
        print(f"--- Шаг h = {h} ---")
        try:
            integralRectangles = rectangles(x0, xk, h)
            integralsRectangles.append(integralRectangles)
            errorRectangles = abs(integral - integralRectangles)
            errorsRectangles.append(errorRectangles)
            print(f"Метод прямоугольников")
            print(f"\tЗначение: {integralRectangles}")
            print(f"\tАбсолютная погрешность (от истинного): {errorRectangles}")
        except ValueError as e:
            print(f"Ошибка при выполнении метода прямоугольников с шагом h={h}: {e}", file=sys.stderr)
        try:
            integralTrapezoids = trapezoids(x0, xk, h)
            integralsTrapezoids.append(integralTrapezoids)
            errorTrapezoids = abs(integral - integralTrapezoids)
            errorsTrapezoids.append(errorTrapezoids)
            print(f"Метод трапеций")
            print(f"\tЗначение: {integralTrapezoids}")
            print(f"\tАбсолютная погрешность (от истинного): {errorTrapezoids}")
        except ValueError as e:
            print(f"Ошибка при выполнении метода трапеций с шагом h={h}: {e}", file=sys.stderr)
        try:
            integralSimpson = simpson(x0, xk, h)
            integralsSimpson.append(integralSimpson)
            errorSimpson = abs(integral - integralSimpson)
            errorsSimpson.append(errorSimpson)
            print(f"Метод Симпсона")
            print(f"\tЗначение: {integralSimpson}")
            print(f"\tАбсолютная погрешность (от истинного): {errorSimpson}")
        except ValueError as e:
            print(f"Ошибка при выполнении метода Симпсона с шагом h={h}: {e}", file=sys.stderr)
        print("==================================================")

    if len(hs) >= 2 and len(integralsRectangles) >= 2 and len(integralsTrapezoids) >= 2 and len(integralsSimpson) >= 2:
        print("--- Уточненные значения по методу Рунге: R_h ≈ (I_h - I_{h/k}) / (1 - k^p)")
        try:
            integralRectanglesRunge = runge(integralsRectangles, hs, orderRectangles)
            errorRectanglesRunge = abs(integral - integralRectanglesRunge)
            runge_error_rectangles_h1 = rungeError(integralsRectangles, [hs[1], hs[0]], orderRectangles) 
            print(f"Метод прямоугольников")
            print(f"\tАпостериорная оценка погрешности по методу Рунге (для шага {hs[1]}): {runge_error_rectangles_h1}")
            print(f"\tУточненное значение: {integralRectanglesRunge}")
            print(f"\tАбсолютная погрешность (от истинного): {errorRectanglesRunge}")
        except ValueError as e:
            print(f"Ошибка при применении метода Рунге для прямоугольников: {e}", file=sys.stderr)
        try:
            integralTrapezoidsRunge = runge(integralsTrapezoids, hs, orderTrapezoids)
            errorTrapezoidsRunge = abs(integral - integralTrapezoidsRunge)
            runge_error_trapezoids_h1 = rungeError(integralsTrapezoids, [hs[1], hs[0]], orderTrapezoids)
            print(f"Метод трапеций")
            print(f"\tАпостериорная оценка погрешности по методу Рунге (для шага {hs[1]}): {runge_error_trapezoids_h1}")
            print(f"\tУточненное значение: {integralTrapezoidsRunge}")
            print(f"\tАбсолютная погрешность (от истинного): {errorTrapezoidsRunge}")
        except ValueError as e:
            print(f"Ошибка при применении метода Рунге для трапеций: {e}", file=sys.stderr)
        try:
            integralSimpsonRunge = runge(integralsSimpson, hs, orderSimpson)
            errorSimpsonRunge = abs(integral - integralSimpsonRunge)
            runge_error_simpson_h1 = rungeError(integralsSimpson, [hs[1], hs[0]], orderSimpson)
            print(f"Метод Симпсона")
            print(f"\tАпостериорная оценка погрешности по методу Рунге (для шага {hs[1]}): {runge_error_simpson_h1}")
            print(f"\tУточненное значение: {integralSimpsonRunge}")
            print(f"\tАбсолютная погрешность (от истинного): {errorSimpsonRunge}")
        except ValueError as e:
            print(f"Ошибка при применении метода Рунге для Симпсона: {e}", file=sys.stderr)
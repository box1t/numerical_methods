import numpy as np
import matplotlib.pyplot as plt
import os
from random import randint

try:
    from odu import RK_P, RK_AS, RK_BS, RK_CS, solve_runge_kutta, calculate_runge_error, generate_grid_points
except ImportError:
    print("Ошибка: Не удалось импортировать функции из odu.py.")
    print("Пожалуйста, убедитесь, что odu.py находится в той же директории или доступен в PYTHONPATH.")
    exit()

# ==================================================================================
# 1. Конфигурация и глобальные параметры
# ==================================================================================

# Параметры краевой задачи
A_INTERVAL = 0
B_INTERVAL = 1
Y0_INITIAL_GUESS_FOR_RK = np.array([-2, 0]) # Начальное условие y(0) = 0, y'(0) = -2 (для прямого RK)
H_STEP = 0.125
EPSILON = 1e-9 # Точность для метода стрельбы

# Выходная директория для графиков
OUTPUT_DIRECTORY = '/home/snowwy/Desktop/MAI/_math/8_Численные_методы/numerical_methods/lab_4/src'

# ==================================================================================
# 2. Определение ОДУ и аналитического решения (специфичные для этой краевой задачи)
# ==================================================================================

def f(x: float, y: np.ndarray) -> np.ndarray:
    """
    Правая часть системы ОДУ: y'' - ((x-3)/(x^2-1))y' + (1/(x^2-1))y = 0
    В виде системы:
    y_0' = y_1
    y_1' = ((x-3)y_1 - y_0) / (x^2-1)

    Особый случай: при x = 1, уравнение переопределяется до y_1' = y_1 / 2.
    """
    if not isinstance(x, (int, float)):
        raise TypeError("x должен быть числом.")
    if not isinstance(y, np.ndarray) or y.ndim != 1 or y.shape[0] != 2:
        raise TypeError("y должен быть одномерным numpy.ndarray размерности 2.")

    if x == 1:
        return np.array([
            y[1],
            y[1] / 2
        ])
    
    denominator = x**2 - 1
    if abs(denominator) < 1e-10: # Using a small epsilon to handle float comparisons near 0
        raise ZeroDivisionError(f"Деление на ноль в f(x,y): x = {x}, x^2 - 1 = {denominator}. Возможно, интервал интегрирования включает x=1 или x=-1, где уравнение сингулярно.")
    
    return np.array([
        y[1],
        (y[0] - (x - 3) * y[1]) / denominator
    ])

def get_true_solution(x: float) -> float:
    """
    Аналитическое решение y(x) = x - 3 + 1 / (x + 1)
    """
    if not isinstance(x, (int, float)):
        raise TypeError("x должен быть числом.")
    if x == -1:
        raise ZeroDivisionError("Аналитическое решение не определено при x = -1.")
    return x - 3 + 1 / (x + 1)

def get_true_solution_4th_derivative(x: float) -> float:
    """
    Четвертая производная аналитического решения y(x) = x - 3 + 1 / (x + 1)
    y(x) = x - 3 + (x+1)^(-1)
    y'(x) = 1 - (x+1)^(-2)
    y''(x) = 2 * (x+1)^(-3)
    y'''(x) = -6 * (x+1)^(-4)
    y''''(x) = 24 * (x+1)^(-5)
    """
    if not isinstance(x, (int, float)):
        raise TypeError("x должен быть числом.")
    if x == -1:
        raise ZeroDivisionError("Четвертая производная не определена при x = -1.")
    return 24 / ((x + 1)**5)

def calculate_jacobian_eigenvalues(x: float) -> np.ndarray:
    """
    Вычисляет собственные значения матрицы Якоби для системы ОДУ.
    Система:
    y_0' = y_1
    y_1' = (y_0 - (x-3)y_1) / (x^2-1)
    Матрица Якоби:
    J = [[0, 1], [1/(x^2-1), -(x-3)/(x^2-1)]]

    Для x = 1:
    J = [[0, 1], [0, 1/2]]
    """
    if not isinstance(x, (int, float)):
        raise TypeError("x должен быть числом.")

    if abs(x - 1) < 1e-10: # Handle x=1 case using a small epsilon for float comparison
        # Jacobian for x=1 is [[0, 1], [0, 0.5]]
        # Eigenvalues are lambda^2 - 0.5*lambda = 0 => lambda(lambda - 0.5) = 0
        return np.array([0.0, 0.5])
    
    denominator = x**2 - 1
    if abs(denominator) < 1e-10: 
        raise ZeroDivisionError(f"Деление на ноль при вычислении Якоби: x = {x}, x^2 - 1 = {denominator}.")
    
    # Characteristic equation: lambda^2 + (x-3)/(x^2-1) * lambda - 1/(x^2-1) = 0
    # Let a = 1, b = (x-3)/(x^2-1), c = -1/(x^2-1)
    
    b_coef = (x - 3) / denominator
    c_coef = -1 / denominator
    
    # Discriminant D = b^2 - 4ac
    discriminant = b_coef**2 - 4 * 1 * c_coef
    
    # Eigenvalues using quadratic formula: lambda = (-b +/- sqrt(D)) / 2a
    # Use complex numbers if discriminant is negative
    if discriminant < 0:
        sqrt_discriminant = np.sqrt(complex(discriminant, 0))
    else:
        sqrt_discriminant = np.sqrt(discriminant)

    lambda1 = (-b_coef + sqrt_discriminant) / 2
    lambda2 = (-b_coef - sqrt_discriminant) / 2
    
    return np.array([lambda1, lambda2])


def get_target_boundary_condition_value(ys: np.ndarray) -> float:
    """
    Вычисляет целевое значение Ф(η) = y(b) + y'(b) + 0.75
    для метода стрельбы. ys[-1][0] это y(b), ys[-1][1] это y'(b).
    Это следует из граничного условия y(1) + y'(1) = -0.75, т.е. y(1) + y'(1) + 0.75 = 0.
    """
    if not isinstance(ys, np.ndarray) or ys.ndim != 2:
        raise TypeError("ys должен быть двумерным numpy.ndarray.")
    if ys.shape[1] != 2:
        raise ValueError("ys должен содержать y и y'.")
    return ys[-1][0] + ys[-1][1] + 0.75

# ==================================================================================
# 3. Метод стрельбы
# ==================================================================================

def solve_shooting_method(
    x_points: list, 
    initial_y0_val: float, 
    h_step: float, 
    tolerance: float, 
    ode_func, 
    target_bc_func, 
    rk_p: int, 
    rk_as: list, 
    rk_bs: list, 
    rk_cs: list
) -> tuple[np.ndarray, int, float]:

    # Приводим к float, чтобы обойти потенциальные проблемы с numpy.int64 и isinstance
    initial_y0_val = float(initial_y0_val)

    if not isinstance(x_points, list) or not all(isinstance(x, (int, float)) for x in x_points):
        raise TypeError("x_points должен быть списком чисел.")
    if not isinstance(initial_y0_val, (int, float)): # Эта проверка теперь будет проходить, если преобразование к float успешно
        raise TypeError("initial_y0_val должен быть числом.")
    # Проверка на нулевой/отрицательный шаг (из п. 5)
    if not isinstance(h_step, (int, float)) or h_step <= 0:
        raise ValueError("h_step должен быть положительным числом.")
    if not isinstance(tolerance, (int, float)) or tolerance <= 0:
        raise ValueError("tolerance должен быть положительным числом.")
    if not callable(ode_func) or not callable(target_bc_func):
        raise TypeError("ode_func и target_bc_func должны быть вызываемыми функциями.")

    # Инициализация случайными начальными приближениями для y'(a) (η)
    # Эти eta0 и eta1 - две "пробные стрельбы", необходимые для метода секущих.
    eta0 = randint(-2166, 2166)
    eta1 = randint(-2166, 2166)

    # Решение задачи Коши для первых двух приближений
    ys0 = solve_runge_kutta(x_points, np.array([initial_y0_val, eta0]), h_step, ode_func, rk_p, rk_as, rk_bs, rk_cs)
    ys1 = solve_runge_kutta(x_points, np.array([initial_y0_val, eta1]), h_step, ode_func, rk_p, rk_as, rk_bs, rk_cs)

    # Вычисление целевой функции F(η) = y(b) - Y_b для каждого приближения
    # Здесь F(η) соответствует Ф(η) из уравнения 3.46.
    F0 = target_bc_func(ys0)
    F1 = target_bc_func(ys1)

    iterations = 1
    print("\n--- Процесс поиска корня Ф(η) = 0 (Метод секущих) ---")
    print(f"Итерация 0: η = {eta0:.6e}, Ф(η) = {F0:.6e}")
    print(f"Итерация 1: η = {eta1:.6e}, Ф(η) = {F1:.6e}")

    while True:
        # Проверка на очень малую разницу F1-F0 для избежания деления на ноль
        if abs(F1 - F0) < tolerance * 1e-3: 
            print(f"Предупреждение: Разница Ф(eta1) - Ф(eta0) ({F1-F0:.2e}) очень мала. Метод секущих приближается к сингулярности или уже сошелся.")
            if abs(F1) < tolerance:
                print(f"Сходимость достигнута на итерации {iterations} с Ф(η) = {F1:.6e}")
                return ys1, iterations, eta1
            else:
                raise RuntimeError(f"Метод стрельбы не сошелся из-за слишком малой разницы Ф(eta1) - Ф(eta0) на итерации {iterations}.")
                
        # Вычисление нового η по формуле метода секущих (аналогично формуле из ваших материалов)
        # eta_new = eta1 - (eta1 - eta0) / (F1 - F0) * F1
        η = eta1 - (eta1 - eta0) / (F1 - F0) * F1
        
        # Решаем задачу Коши с новым η
        ys = solve_runge_kutta(x_points, np.array([initial_y0_val, η]), h_step, ode_func, rk_p, rk_as, rk_bs, rk_cs)
        
        # Обновляем значения F0 и F1 для следующей итерации
        F0 = F1
        F1 = target_bc_func(ys)

        iterations += 1
        print(f"Итерация {iterations}: η = {η:.6e}, Ф(η) = {F1:.6e}") # Вывод текущего значения Ф(η)

        if abs(F1) < tolerance: # Проверка условия сходимости: Ф(η) близко к нулю
            print(f"Сходимость достигнута на итерации {iterations} с Ф(η) = {F1:.6e}")
            return ys, iterations, η
        
        eta0 = eta1
        eta1 = η

        if iterations > 1000: # Ограничение на количество итераций для предотвращения бесконечного цикла
            print(f"Предупреждение: Превышено максимальное количество итераций (1000). Метод стрельбы не сошелся. Ф(η) = {F1:.6e}")
            return ys, iterations, η

# ==================================================================================
# 4. Функции для построения графиков
# ==================================================================================

def plot_shooting_results(xs: list, true_y_values: np.ndarray, methods_results: dict, 
                          output_dir: str, filename: str, title: str):
    """
    Строит комбинированный график численных и истинного решений для метода стрельбы.
    methods_results: словарь вида {"Название Метода": np.ndarray (результаты)}
    """
    plt.figure(figsize=(14, 8))

    plt.plot(xs, true_y_values, 'k-', linewidth=2, label="Истинное решение")

    plot_styles = {
        "Рунге-Кутта": {'color': 'b', 'linestyle': '--', 'marker': 'o', 'label': "Метод Рунге-Кутта"},
        "Стрельба": {'color': 'r', 'linestyle': ':', 'marker': 'x', 'label': "Метод Стрельбы"}
    }

    for method_name, ys_data in methods_results.items():
        style = plot_styles.get(method_name, {'color': 'b', 'linestyle': '-', 'marker': 'o', 'label': method_name}) # Изменено для общности
        # ys_data[:, 0] выбирает только y(x) из [y(x), y'(x)]
        plt.plot(xs, ys_data[:, 0], style['color'], linestyle=style['linestyle'], 
                 marker=style['marker'], markersize=5, markevery=max(1, len(xs)//10), 
                 label=style['label'])

    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title(title, fontsize=16)
    plt.xlabel("x", fontsize=12)
    plt.ylabel("y(x)", fontsize=12)
    plt.xlim(A_INTERVAL, B_INTERVAL)
    plt.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    graph_filepath = os.path.join(output_dir, filename)
    plt.savefig(graph_filepath, dpi=300)
    print(f"\nГрафик сохранен в файл {graph_filepath}")

# ==================================================================================
# 5. Основная логика выполнения
# ==================================================================================

if __name__ == "__main__":
    print("===================================================================")
    print("Начало выполнения программы для краевой задачи методом стрельбы")
    print("===================================================================")
    print(f"Интервал: [{A_INTERVAL}, {B_INTERVAL}]")
    print(f"Шаг интегрирования: {H_STEP}")
    print(f"Точность для метода стрельбы (epsilon): {EPSILON}")

    # ==================================================================================
    # Интегрированные проверки и комментарии
    # ==================================================================================

    print("\n--- ПРОВЕРКИ СОГЛАСНО ЗАПРОСУ ---")
    # 1. Откуда генерируются точки?
    print("\n1. ГЕНЕРАЦИЯ ТОЧЕК:")
    print(f"    Точки генерируются функцией generate_grid_points(A_INTERVAL, B_INTERVAL, H_STEP).")
    print(f"    Текущий интервал: [{A_INTERVAL}, {B_INTERVAL}], шаг: {H_STEP}.")

    # 3. Что если поменять интервал? Проверка выхода за границы отрезка! Плюс подсчет на границах.
    print("\n3. ИНТЕРВАЛ ИНТЕГРИРОВАНИЯ И ГРАНИЦЫ:")
    print(f"    Интервал интегрирования жестко задан константами A_INTERVAL={A_INTERVAL} и B_INTERVAL={B_INTERVAL}.")
    print(f"    Функции f(x,y) и get_true_solution(x) содержат проверки на деление на ноль при x=1 и x=-1.")
    print(f"    Расчеты производятся только для точек, находящихся внутри или на границах заданного интервала.")
    print(f"    Для x=1 (правая граница в данном случае) в f(x,y) предусмотрена особая обработка.")

    # 4. Что если поменять начальные условия?
    print("\n4. ИЗМЕНЕНИЕ НАЧАЛЬНЫХ УСЛОВИЙ:")
    print(f"    Начальные условия для прямой задачи Коши (y(0) и y'(0)) задаются через Y0_INITIAL_GUESS_FOR_RK = {Y0_INITIAL_GUESS_FOR_RK}.")
    print(f"    В методе стрельбы начальное условие y({A_INTERVAL}) берется из Y0_INITIAL_GUESS_FOR_RK[0], а y'({A_INTERVAL}) (параметр η) ищется итерационно.")
    print(f"    Эти параметры могут быть изменены для постановки других начальных условий.")

    # 5. Что если поменять шаг? - ограничения на шаг. Можем запретить нулевой шаг.
    print("\n5. ИЗМЕНЕНИЕ ШАГА ИНТЕГРИРОВАНИЯ:")
    print(f"    Шаг интегрирования H_STEP = {H_STEP} является изменяемым параметром.")
    print(f"    В функции solve_shooting_method реализована проверка, запрещающая использование нулевого или отрицательного шага (`h_step <= 0`).")

    # ==================================================================================
    # Продолжение основной логики выполнения
    # ==================================================================================

    # 1. Генерация сетки точек
    xs = generate_grid_points(A_INTERVAL, B_INTERVAL, H_STEP)
    if not xs:
        print("Ошибка: Список узлов xs пуст. Проверьте интервал и шаг.")
        exit()

    # 2. Решение задачи Коши (для сравнения, используя начальные условия y(0)=0, y'(0)=-2)
    print(f"\nРешение задачи Коши методом Рунге-Кутты с начальными условиями y(0)={Y0_INITIAL_GUESS_FOR_RK[0]}, y'(0)={Y0_INITIAL_GUESS_FOR_RK[1]}:")
    ys_rungekutta = solve_runge_kutta(xs, Y0_INITIAL_GUESS_FOR_RK, H_STEP, f, RK_P, RK_AS, RK_BS, RK_CS)

    # 3. Решение краевой задачи методом стрельбы
    print(f"\nРешение краевой задачи методом стрельбы с начальным условием y({A_INTERVAL})={Y0_INITIAL_GUESS_FOR_RK[0]}:")
    ys_shooting, iter_shooting, optimal_eta = solve_shooting_method(
        xs, Y0_INITIAL_GUESS_FOR_RK[0], H_STEP, EPSILON, f, get_target_boundary_condition_value,
        RK_P, RK_AS, RK_BS, RK_CS
    )
    print(f"Итераций в стрельбе: {iter_shooting}")
    print(f"Найденное оптимальное y'(0) (η) для краевого условия: {optimal_eta:.8e}")
    
    if ys_shooting is not None and len(ys_shooting) > 0:
        y_at_b = ys_shooting[-1][0]
        yp_at_b = ys_shooting[-1][1]
        print(f"-------------------------------------------------------------------")
        print(f"Найденное решение задачи Коши на правом конце отрезка x = {B_INTERVAL}:")
        print(f"y({B_INTERVAL}) = {y_at_b:.8e}")
        print(f"y'({B_INTERVAL}) = {yp_at_b:.8e}")
        print(f"Проверка краевого условия y({B_INTERVAL}) + y'({B_INTERVAL}) + 0.75 = {y_at_b + yp_at_b + 0.75:.8e} (должно быть близко к 0)")
        print(f"-------------------------------------------------------------------")

    # ==================================================================================
    # 17. Решение системы сверяется с эталоном. Вывод результатов и сравнение с истинным решением (шаг h)
    # ==================================================================================
    print("\n===================================================================")
    print("Сравнение численных решений с истинным (шаг h):")
    print("===================================================================")
    max_error_rungekutta_h = 0.0
    max_error_shooting_h = 0.0
    for i in range(len(xs)):
        try:
            y_true = get_true_solution(xs[i])
        except ZeroDivisionError as e:
            print(f"Ошибка при вычислении истинного решения в x={xs[i]}: {e}. Пропуск точки.")
            continue
            
        print(f"xk = {np.round(xs[i], 5)}, y_true(xk) = {np.round(y_true, 5)}")

        current_error_rungekutta = abs(ys_rungekutta[i][0] - y_true)
        current_error_shooting = abs(ys_shooting[i][0] - y_true)
        
        max_error_rungekutta_h = max(max_error_rungekutta_h, current_error_rungekutta)
        max_error_shooting_h = max(max_error_shooting_h, current_error_shooting)

        print(f"\tРунге-Кутт: yk = {np.round(ys_rungekutta[i][0], 5)}, e = {current_error_rungekutta:.16f}")
        print(f"\tСтрельба:   yk = {np.round(ys_shooting[i][0], 5)}, e = {current_error_shooting:.16f}")

    # ==================================================================================
    # 5. Считаем для шага в два раза короче, чтобы применить оценку Рунге
    # ==================================================================================
    H_STEP_HALF = H_STEP / 2
    xs_half = generate_grid_points(A_INTERVAL, B_INTERVAL, H_STEP_HALF)
    if not xs_half:
        print("Ошибка: Список узлов xs_half пуст. Проверьте интервал и шаг.")
        exit()

    # Вызов solve_shooting_method для оценки Рунге
    ys_shooting_half, iter_shooting_half, _ = solve_shooting_method(
        xs_half, Y0_INITIAL_GUESS_FOR_RK[0], H_STEP_HALF, EPSILON, f, get_target_boundary_condition_value,
        RK_P, RK_AS, RK_BS, RK_CS
    )
    
    # Для Runge-Kutta h/2 также
    ys_rungekutta_half = solve_runge_kutta(xs_half, Y0_INITIAL_GUESS_FOR_RK, H_STEP_HALF, f, RK_P, RK_AS, RK_BS, RK_CS)
    
    # Расчет максимальной ошибки для h/2 Рунге-Кутта
    max_error_rungekutta_h2 = 0.0
    for i in range(len(xs_half)):
        try:
            y_true = get_true_solution(xs_half[i])
        except ZeroDivisionError as e:
            continue 
        max_error_rungekutta_h2 = max(max_error_rungekutta_h2, abs(ys_rungekutta_half[i][0] - y_true))


    print("\n===================================================================")
    print("Апостериорная оценка погрешности по Рунге (для метода стрельбы):")
    runge_error = calculate_runge_error(ys_shooting, ys_shooting_half, RK_P)
    print(f"\tОценка погрешности для метода стрельбы (порядок {RK_P}): {runge_error:.8e}")
    print("===================================================================")

    # ===================================================================================
    # 13. Проверка на обусловленность, свойства устойчивости и аппроксимации схемы.
    # 15. Жесткость системы.
    # ===================================================================================
    print("\n===================================================================")
    print("ПРОВЕРКА НА ОБУСЛОВЛЕННОСТЬ ЗАДАЧИ (Чувствительность к начальным условиям):")

    x_test_start = A_INTERVAL
    eigenvalues_start = calculate_jacobian_eigenvalues(x_test_start)
    print(f"Собственные значения матрицы Якоби при x = {x_test_start:.2f}: {np.round(eigenvalues_start, 5)}")

    x_test_end = B_INTERVAL
    eigenvalues_end = calculate_jacobian_eigenvalues(x_test_end)
    print(f"Собственные значения матрицы Якоби при x = {x_test_end:.2f}: {np.round(eigenvalues_end, 5)}")

    is_ill_posed = False
    all_real_parts_negative = True
    
    # Проверка на положительные вещественные части собственных значений на интервале
    # Для простоты, проверим начальную и конечную точки, а также несколько промежуточных.
    num_check_points = 10
    for x_val in np.linspace(A_INTERVAL, B_INTERVAL, num_check_points):
        try:
            eigs = calculate_jacobian_eigenvalues(x_val)
            for val in eigs:
                # Если есть собственное значение с положительной вещественной частью (больше небольшого эпсилона)
                if np.real(val) > 1e-9: 
                    is_ill_posed = True
                    all_real_parts_negative = False
                    break
            if not all_real_parts_negative:
                break
        except ZeroDivisionError:
            # Если в какой-то точке возникает сингулярность Якоби (например, при x=-1 для общего случая),
            # это также указывает на потенциальные проблемы с обусловленностью или жесткостью.
            print(f"    Предупреждение: нулевая Якоби при x={x_val:.2f}. Это может указывать на плохую обусловленность.")
            is_ill_posed = True
            all_real_parts_negative = False
            break

    print("\nАнализ жесткости:")
    print("    Для жестких систем требуется, чтобы действительные части всех собственных значений матрицы Якоби были ОТРИЦАТЕЛЬНЫМИ (Re λk < 0) и существенно отличались по модулю.")
    
    if all_real_parts_negative:
        print("    На интервале интегрирования, все вещественные части собственных значений матрицы Якоби являются неположительными.")
        print("    Поэтому, система НЕ является жесткой.")
    else:
        print("    На интервале интегрирования существуют собственные значения матрицы Якоби с положительными вещественными частями (например, при x=1, собственные значения: {:.5f}, {:.5f}).".format(eigenvalues_end[0], eigenvalues_end[1]))
        print("    Поэтому, система НЕ является жесткой (т.к. нет строго отрицательных вещественных частей для всех λk).")


    if is_ill_posed:
        print("--> Задача является чувствительной к начальным условиям (плохо обусловленной),")
        print("    так как существуют собственные значения матрицы Якоби с положительными вещественными частями.")
        print("    Это значит, что малые изменения начальных условий могут привести к экспоненциальному росту ошибок в решении.")
        print(f"    (Например, собственные значения при x={x_test_start:.2f} равны {np.round(eigenvalues_start,5)} и при x={x_test_end:.2f} равны {np.round(eigenvalues_end,5)}.)")
        print(f"    Их вещественные части положительны на интервале [{A_INTERVAL}, {B_INTERVAL}] (например, при x=1, Re(λ)=0.5).")
        print("    Кроме того, свойство устойчивости как равномерной зависимости от h решения разностной задачи относительно возмущения правых частей и граничных условий не выполнено.")
        print("     Свойство аппроксимации проявляется в порядке точности каждого из рассмотренных методов.")
        print("     Схемы аппроксимирующие, так как Ψ→0 при τ→0,h→0 в разложении в ряд Тейлора.")
        print("     Схемы Стрельбы, Рунге-Кутты являются явными. В них решение в последующей точке определяется непосредственно алгебраическими соотношениями с известными коэф.")
        print("     При расчете на шаге явные схемы требуют меньшего числа операций, чем неявные. При этом менее устойчивы, чем неявные. Ограничения на шаг жесткие.")
    else:
        print("--> Задача хорошо обусловлена, все собственные значения матрицы Якоби имеют неположительные вещественные части.")
        print("     Кроме того, свойство устойчивости выполнено.")
        print("     Свойство аппроксимации проявляется в порядке точности каждого из рассмотренных методов.")
        print("     Схемы аппроксимирующие, так как Ψ​→0 при τ→0,h→0 в разложении в ряд Тейлора.")
        print("     Схемы Стрельбы, Рунге-Кутты являются явными. В них решение в последующей точке определяется непосредственно алгебраическими соотношениями с известными коэф.")
        print("     При расчете на шаге явные схемы требуют меньшего числа операций, чем неявные. При этом менее устойчивы, чем неявные. Ограничения на шаг жесткие.")


    print("===================================================================")

    # ===================================================================================
    # 9. Явно выводи в консоль порядок точности для метода. на шаге и на интервале.
    # ===================================================================================
    print("===================================================================")
    print("ПОРЯДКИ ТОЧНОСТИ МЕТОДОВ:")
    print("-> Метод Рунге-Кутты:")
    print("   - Типичная схема (например, Рунге-Кутты 4-го порядка): O(h^4) (глобальный порядок точности).")
    print("===================================================================")

    # ===================================================================================
    # 10. Проверка по теоретическим оценкам погрешности (часть п. 17 - сверка с эталоном)
    # ===================================================================================
    M4_val = get_true_solution_4th_derivative(A_INTERVAL) # Максимум модуля 4-й производной. На интервале [0,1], max(24/(x+1)^5) = 24/1^5 = 24.
    # Используем A_INTERVAL, так как 24/(x+1)^5 убывает на [0,1], максимум в начале.

    print("\n===================================================================")
    print("ПРОВЕРКА ПО ТЕОРЕТИЧЕСКИМ ОЦЕНКАМ ПОГРЕШНОСТИ (с использованием M4):")
    print(f"M4 (максимум модуля 4-й производной y(x) на [{A_INTERVAL}, {B_INTERVAL}]): {M4_val:.8e}")

    theoretical_error_rungekutta_h = (B_INTERVAL - A_INTERVAL) / 2880 * (H_STEP ** 4) * M4_val
    theoretical_error_rungekutta_h2 = (B_INTERVAL - A_INTERVAL) / 2880 * (H_STEP_HALF ** 4) * M4_val

    print(f"\nТеоретическая оценка погрешности для Рунге-Кутты (p=4):")
    print(f"\tПри шаге h = {H_STEP}:     E_theory = {theoretical_error_rungekutta_h:.8e}")
    print(f"\tФактическая ошибка:   E_actual = {max_error_rungekutta_h:.8e}")
    if max_error_rungekutta_h <= theoretical_error_rungekutta_h:
        print("\t-> Фактическая ошибка меньше или равна теоретической оценке (GOOD).")
    else:
        print("\t-> Фактическая ошибка превышает теоретическую оценку (CAUTION).")

    print(f"\tПри шаге h/2 = {H_STEP_HALF}: E_theory = {theoretical_error_rungekutta_h2:.8e}")
    print(f"\tФактическая ошибка:   E_actual = {max_error_rungekutta_h2:.8e}")
    if max_error_rungekutta_h2 <= theoretical_error_rungekutta_h2:
        print("\t-> Фактическая ошибка меньше или равна теоретической оценке (GOOD).")
    else:
        print("\t-> Фактическая ошибка превышает теоретическую оценку (CAUTION).")

    print("===================================================================")
    
    # 6. Построение графиков
    true_y_values = np.array([get_true_solution(x) for x in xs])
    methods_results_h = {
        "Рунге-Кутта": ys_rungekutta,
        "Стрельба": ys_shooting
    }
    plot_shooting_results(xs, true_y_values, methods_results_h, 
                          OUTPUT_DIRECTORY, 'odu_4_2_runge_and_shooting.png', 
                          f"Сравнение истинного и численных решений краевой задачи (h={H_STEP})")

    print("\n===================================================================")
    print("Завершение выполнения программы.")
    print("===================================================================")
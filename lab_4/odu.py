import numpy as np
import matplotlib.pyplot as plt
import os

# ==================================================================================
# 1. Конфигурация и глобальные параметры
# ==================================================================================

# Параметры задачи
A_INTERVAL = 1
B_INTERVAL = 2
Y0_INITIAL = np.array([2 + np.e, 1 + np.e])
H_STEP = 0.1

# Параметры метода Рунге-Кутты (4-й порядок)
RK_P = 4
RK_AS = [0, 0.5, 0.5, 1]
RK_BS = [[0.5], [0, 0.5], [0, 0, 1]]
RK_CS = [1/6, 1/3, 1/3, 1/6]

# Параметры отладки
DEBUG_MODE = False

# Выходная директория для графиков
OUTPUT_DIRECTORY = '/home/snowwy/Desktop/MAI/_math/8_Численные_методы/numerical_methods/lab_4/src'

# ==================================================================================
# 2. Определение ОДУ и аналитического решения
# ==================================================================================

def f(x: float, y: np.ndarray) -> np.ndarray:
    """
    Правая часть системы ОДУ: y'' - ((x+1)/x)y' + (1/x)y = 0
    В виде системы:
    y_0' = y_1
    y_1' = ((x+1)/x)y_1 - (1/x)y_0
    """
    if not isinstance(x, (int, float)):
        raise TypeError("x должен быть числом.")
    if not isinstance(y, np.ndarray) or y.ndim != 1 or y.shape[0] != 2:
        raise TypeError("y должен быть одномерным numpy.ndarray размерности 2.")

    if x == 0:
        raise ZeroDivisionError("Деление на ноль в f(x,y): x не может быть равно 0.")
    
    return np.array([
        y[1],
        ((x + 1) * y[1] - y[0]) / x
    ])

def get_true_solution(x: float) -> float:
    """
    Аналитическое решение y(x) = x + 1 + e^x
    """
    if not isinstance(x, (int, float)):
        raise TypeError("x должен быть числом.")
    return x + 1 + np.exp(x)

def get_true_solution_4th_derivative(x: float) -> float:
    """
    Четвертая производная аналитического решения y(x) = x + 1 + e^x
    y''''(x) = e^x
    """
    if not isinstance(x, (int, float)):
        raise TypeError("x должен быть числом.")
    return np.exp(x)

# ==================================================================================
# 3. Вспомогательные функции
# ==================================================================================

def validate_initial_parameters(a: float, b: float, y0: np.ndarray, h: float):
    """
    Проверяет корректность входных параметров.
    """
    if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        raise TypeError("Границы интервала 'a' и 'b' должны быть числами.")
    if a >= b:
        raise ValueError(f"Левая граница интервала 'a' ({a}) должна быть строго меньше правой границы 'b' ({b}).")

    if not isinstance(y0, np.ndarray):
        raise TypeError("Начальные условия 'y0' должны быть numpy.ndarray.")
    if y0.ndim != 1:
        raise ValueError("Начальные условия 'y0' должны быть одномерным массивом (вектором).")
    if y0.shape[0] != 2:
        raise ValueError(f"Размерность вектора начальных условий 'y0' ({y0.shape[0]}) не соответствует размерности системы ОДУ (2).")

    if not (isinstance(h, (int, float))):
        raise TypeError("Шаг интегрирования 'h' должен быть числом.")
    if h <= 0:
        raise ValueError(f"Шаг интегрирования 'h' ({h}) должен быть строго положительным числом.")

    if a <= 0 <= b:
        print(f"ПРЕДУПРЕЖДЕНИЕ: Интервал интегрирования [{a}, {b}] включает или касается x=0.")
        print("                 Функция f(x,y) содержит деление на x, что может привести к ошибке.")
        if a < 0 < b:
            raise ValueError(f"Интервал интегрирования [{a}, {b}] содержит x=0, где функция f(x,y) не определена (деление на x).")
        elif a == 0:
            raise ValueError(f"Начальная точка интервала x=0, где функция f(x,y) не определена (деление на x).")

def generate_grid_points(x0: float, xk: float, h: float) -> list:
    """
    Генерирует равномерную сетку точек на интервале [x0, xk] с шагом h.
    """
    if not (isinstance(x0, (int, float)) and isinstance(xk, (int, float)) and isinstance(h, (int, float))):
        raise TypeError("Аргументы generate_grid_points должны быть числами.")
    if h <= 0:
        raise ValueError("Шаг h должен быть положительным числом.")
    if x0 >= xk:
        raise ValueError("x0 должно быть меньше xk для создания интервала.")

    xs = []
    x = x0
    while x < xk + h * 0.5:
        xs.append(x)
        x += h
    if not xs or abs(xs[-1] - xk) > h * 0.5:
        xs.append(xk)
    
    if len(xs) > 1 and abs(xs[-1] - xs[-2]) < 1e-9: 
        xs.pop()

    if len(xs) > 2:
        first_diff = xs[1] - xs[0]
        for i in range(2, len(xs)):
            current_diff = xs[i] - xs[i-1]
            if abs(current_diff - first_diff) > 1e-9 * first_diff:
                print(f"ПРЕДУПРЕЖДЕНИЕ: Сетка может быть неравномерной. Разница шагов: {first_diff:.8e}, {current_diff:.8e}")
                break
    
    return xs

def calculate_runge_error(ys_h: np.ndarray, ys_h2: np.ndarray, p: int, k: int = 2) -> float:
    """
    Вычисляет апостериорную оценку погрешности по Рунге.
    ys_h: решение с шагом h
    ys_h2: решение с шагом h/k (k=2 по умолчанию)
    p: порядок точности метода
    """
    if not isinstance(ys_h, np.ndarray) or not isinstance(ys_h2, np.ndarray):
        raise TypeError("ys_h и ys_h2 должны быть numpy.ndarray.")
    if ys_h.ndim != 2 or ys_h2.ndim != 2:
        raise ValueError("ys_h и ys_h2 должны быть двумерными массивами (N_points x dim).")
    if not isinstance(p, int) or p <= 0:
        raise ValueError("Порядок точности p должен быть положительным целым числом.")
    if abs(p - round(p)) > 1e-9:
        print(f"ПРЕДУПРЕЖДЕНИЕ: Порядок точности p={p} не является целым числом. Это может привести к некорректной оценке.")

    error = 0
    
    expected_len_ys2 = len(ys_h) * k
    if not (abs(len(ys_h2) - expected_len_ys2) <= 2):
         print(f"ПРЕДУПРЕЖДЕНИЕ: Размеры массивов ys_h ({len(ys_h)}) и ys_h2 ({len(ys_h2)}) не соответствуют ожидаемому соотношению {k}:1 для оценки Рунге.")
         print("                 Это может указывать на проблему в генерации сетки или несовместимость интервалов.")

    for i in range(ys_h.shape[0]):
        if (i * k) >= ys_h2.shape[0]:
            print(f"ПРЕДУПРЕЖДЕНИЕ: Индекс {i * k} выходит за пределы ys_h2 (размер {ys_h2.shape[0]}) при расчете ошибки Рунге. Пропуск оставшихся точек.")
            break

        denominator = (k ** p - 1)
        if abs(denominator) < 1e-10:
            raise ValueError(f"Знаменатель (k^p - 1) в формуле Рунге очень близок к нулю (k={k}, p={p}). Это означает, что p не соответствует порядку k, или p слишком мало.")
            
        error = max(error, abs(ys_h2[i * k][0] - ys_h[i][0]) / denominator)
    return error

def calculate_max_absolute_error(xs_vals: list, ys_vals: np.ndarray, true_solution_func) -> float:
    """
    Вычисляет максимальную абсолютную ошибку между численным и аналитическим решением.
    """
    if not isinstance(xs_vals, list) or not all(isinstance(x, (int, float)) for x in xs_vals):
        raise TypeError("xs_vals должен быть списком чисел.")
    if not isinstance(ys_vals, np.ndarray) or ys_vals.ndim != 2:
        raise TypeError("ys_vals должен быть двумерным numpy.ndarray.")
    if len(xs_vals) != ys_vals.shape[0]:
        raise ValueError("Размерность xs_vals и ys_vals не совпадают.")
    if not callable(true_solution_func):
        raise TypeError("true_solution_func должна быть вызываемой функцией.")

    max_error = 0
    for i in range(len(xs_vals)):
        true_y = true_solution_func(xs_vals[i])
        current_error = abs(ys_vals[i][0] - true_y)
        if current_error > max_error:
            max_error = current_error
    return max_error

def calculate_jacobian_eigenvalues(x_val: float) -> np.ndarray:
    """
    Вычисляет собственные значения матрицы Якоби для данной системы ОДУ в точке x.
    """
    if x_val == 0:
        raise ZeroDivisionError("Невозможно вычислить Якобиан при x=0, так как функция не определена.")
    
    jacobian_matrix = np.array([
        [0, 1],
        [-1/x_val, (x_val + 1)/x_val]
    ])
    
    eigenvalues = np.linalg.eigvals(jacobian_matrix)
    return eigenvalues

# ==================================================================================
# 4. Численные методы решения ОДУ
# ==================================================================================

def _get_runge_kutta_Ks(x: float, y: np.ndarray, h: float, ode_func, p: int, as_: list, bs: list) -> np.ndarray:
    """
    Внутренняя вспомогательная функция для расчета коэффициентов K в методе Рунге-Кутты.
    """
    dim = y.shape[0]
    Ks = np.empty((p, dim))

    for i in range(p):
        newX = x + as_[i] * h
        newY = np.copy(y)
        for j in range(i):
            newY += bs[i - 1][j] * Ks[j]

        # if newX == 0:
        #      raise ZeroDivisionError(f"Деление на ноль в ode_func(x,y): промежуточная точка x = {newX} при расчете K{i+1}.")
        
        K = h * ode_func(newX, newY)
        if DEBUG_MODE: print(f"\tK{i + 1} = {K}")
        Ks[i] = K

    return Ks

def _get_runge_kutta_deltaY(x: float, y: np.ndarray, h: float, ode_func, p: int, as_: list, bs: list, cs: list) -> np.ndarray:
    """
    Внутренняя вспомогательная функция для расчета приращения DeltaY в методе Рунге-Кутты.
    """
    Ks = _get_runge_kutta_Ks(x, y, h, ode_func, p, as_, bs)
    dim = Ks.shape[1]
    sum_ = np.zeros(dim)
    for i in range(p):
        sum_ += cs[i] * Ks[i]
    if DEBUG_MODE: print(f"\tdeltaY = {sum_}")
    return sum_

def solve_runge_kutta(xs: list, y0: np.ndarray, h: float, ode_func, p: int, as_: list, bs: list, cs: list) -> np.ndarray:
    """
    Решает ОДУ методом Рунге-Кутты 4-го порядка.
    """
    if not isinstance(xs, list) or not all(isinstance(x, (int, float)) for x in xs):
        raise TypeError("xs должен быть списком чисел.")
    if not isinstance(y0, np.ndarray) or y0.ndim != 1:
        raise TypeError("y0 должен быть одномерным numpy.ndarray.")
    if not isinstance(h, (int, float)) or h <= 0:
        raise ValueError("h должен быть положительным числом.")
    if len(xs) < 1:
        raise ValueError("Список узлов xs не должен быть пустым.")
    
    N = len(xs) - 1
    dim = y0.shape[0]
    ys = np.empty((N + 1, dim))
    ys[0] = y0

    if DEBUG_MODE: print(f"N = {N}, dim = {dim}")

    for k in range(1, N + 1):
        if DEBUG_MODE: print(f"Шаг {k}")
        current_x_point = xs[k - 1]
        # Дополнительная проверка, если `a` и `b` были бы параметрами функции
        # if not (a_interval <= current_x_point <= b_interval + 1e-9):
        #      print(f"ПРЕДУПРЕЖДЕНИЕ: Расчетная точка x={current_x_point:.5f} находится вне заданного интервала [{a_interval}, {b_interval}].")

        ys[k] = ys[k - 1] + _get_runge_kutta_deltaY(xs[k - 1], ys[k - 1], h, ode_func, p, as_, bs, cs)
        
        if np.isnan(ys[k]).any() or np.isinf(ys[k]).any():
            raise FloatingPointError(f"Обнаружены NaN или Inf в решении Рунге-Кутты на шаге {k}. Проверьте ode_func(x,y) или уменьшите шаг.")

        if DEBUG_MODE: print(f"\ty = {ys[k]}")

    return ys


def solve_euler(xs: list, y0: np.ndarray, h: float, ode_func) -> np.ndarray:
    """
    Решает ОДУ методом Эйлера.
    """
    if not isinstance(xs, list) or not all(isinstance(x, (int, float)) for x in xs):
        raise TypeError("xs должен быть списком чисел.")
    if not isinstance(y0, np.ndarray) or y0.ndim != 1:
        raise TypeError("y0 должен быть одномерным numpy.ndarray.")
    if not isinstance(h, (int, float)) or h <= 0:
        raise ValueError("h должен быть положительным числом.")
    if len(xs) < 1:
        raise ValueError("Список узлов xs не должен быть пустым.")
    
    N = len(xs) - 1
    dim = y0.shape[0]
    ys = np.empty((N + 1, dim))
    ys[0] = y0

    for k in range(N):
        current_x_point = xs[k]
        # if not (a_interval <= current_x_point <= b_interval + 1e-9):
        #      print(f"ПРЕДУПРЕЖДЕНИЕ: Расчетная точка x={current_x_point:.5f} находится вне заданного интервала [{a_interval}, {b_interval}].")

        ys[k + 1] = ys[k] + h * ode_func(xs[k], ys[k])
        
        if np.isnan(ys[k+1]).any() or np.isinf(ys[k+1]).any():
            raise FloatingPointError(f"Обнаружены NaN или Inf в решении Эйлера на шаге {k+1}. Проверьте ode_func(x,y) или уменьшите шаг.")

    return ys


def solve_adams(xs: list, y0_initial_points: np.ndarray, h: float, ode_func, required_initial_points: int = 4) -> np.ndarray:
    """
    Решает ОДУ методом Адамса (4-го порядка).
    y0_initial_points: numpy.ndarray, содержащий начальные точки, необходимые для инициализации метода.
                       Обычно получаются другим методом, например, Рунге-Куттой.
    """
    if not isinstance(xs, list) or not all(isinstance(x, (int, float)) for x in xs):
        raise TypeError("xs должен быть списком чисел.")
    if not isinstance(y0_initial_points, np.ndarray) or y0_initial_points.ndim != 2:
        raise TypeError("y0_initial_points должен быть двумерным numpy.ndarray (N_points x dim).")
    if not isinstance(h, (int, float)) or h <= 0:
        raise ValueError("h должен быть положительным числом.")
    if len(xs) < 1:
        raise ValueError("Список узлов xs не должен быть пустым.")

    if y0_initial_points.shape[0] < required_initial_points:
        raise ValueError(f"Метод Адамса 4-го порядка требует как минимум {required_initial_points} начальные точки (y0_initial_points) для инициализации. Получено {y0_initial_points.shape[0]}.")
    if y0_initial_points.shape[1] != Y0_INITIAL.shape[0]: # Use global Y0_INITIAL for dim check
        raise ValueError(f"Размерность векторов в y0_initial_points ({y0_initial_points.shape[1]}) не соответствует размерности системы ОДУ ({Y0_INITIAL.shape[0]}).")

    N = len(xs) - 1
    dim = y0_initial_points.shape[1]
    ys = np.empty((N + 1, dim))
    
    fs = np.empty((N + 1, dim))

    for i in range(required_initial_points):
        ys[i] = np.copy(y0_initial_points[i])
        fs[i] = ode_func(xs[i], ys[i])

    for k in range(required_initial_points, N + 1):
        current_x_point = xs[k - 1]
        # if not (a_interval <= current_x_point <= b_interval + 1e-9):
        #      print(f"ПРЕДУПРЕЖДЕНИЕ: Расчетная точка x={current_x_point:.5f} находится вне заданного интервала [{a_interval}, {b_interval}].")
        
        ys[k] = ys[k - 1] + h/24 * (55 * fs[k - 1] - 59 * fs[k - 2] + 37 * fs[k - 3] - 9 * fs[k - 4])
        fs[k] = ode_func(xs[k], ys[k]) # Calculate f for the newly found y_k

        if np.isnan(ys[k]).any() or np.isinf(ys[k]).any():
            raise FloatingPointError(f"Обнаружены NaN или Inf в решении Адамса на шаге {k}. Проверьте ode_func(x,y) или уменьшите шаг.")

    return ys

# ==================================================================================
# 5. Функции для построения графиков
# ==================================================================================

def plot_results_combined(xs: list, true_y_values: np.ndarray, methods_results: dict, 
                          output_dir: str, filename: str, title: str):
    """
    Строит комбинированный график численных и истинного решений.
    methods_results: словарь вида {"Название Метода": np.ndarray (результаты)}
    """
    plt.figure(figsize=(14, 8))

    plt.plot(xs, true_y_values, 'k-', linewidth=2, label="Истинное решение")

    plot_styles = {
        "Эйлер": {'color': 'g', 'linestyle': ':', 'marker': 's', 'label': "Метод Эйлера"},
        "Рунге-Кутта": {'color': 'b', 'linestyle': '--', 'marker': 'o', 'label': "Метод Рунге-Кутта"},
        "Адамс": {'color': 'r', 'linestyle': '-.', 'marker': 'x', 'label': "Метод Адамса"}
    }

    for method_name, ys_data in methods_results.items():
        style = plot_styles.get(method_name, {'color': 'c', 'linestyle': '-', 'marker': '.', 'label': method_name})
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
    print(f"\nОбщий график сохранен в файл {graph_filepath}")

def plot_results_individual(xs_h: list, ys_h: np.ndarray, xs_h2: list, ys_h2: np.ndarray, 
                            true_solution_func, output_dir: str, method_name: str, h: float, h2: float):
    """
    Строит отдельные графики для каждого метода, сравнивая решения с шагами h и h/2.
    """
    plt.figure(figsize=(14, 8))
    
    true_y_values_h2 = np.array([true_solution_func(x) for x in xs_h2])
    plt.plot(xs_h2, true_y_values_h2, 'k-', linewidth=2, label="Истинное решение")
    
    plot_styles_h = {
        "Эйлер": {'color': 'g', 'linestyle': ':', 'marker': 's'},
        "Рунге-Кутта": {'color': 'b', 'linestyle': ':', 'marker': 'o'},
        "Адамс": {'color': 'r', 'linestyle': ':', 'marker': 'x'}
    }
    plot_styles_h2 = {
        "Эйлер": {'color': 'g', 'linestyle': '--', 'marker': '^'},
        "Рунге-Кутта": {'color': 'b', 'linestyle': '--', 'marker': 'v'},
        "Адамс": {'color': 'r', 'linestyle': '--', 'marker': 'P'}
    }

    style_h = plot_styles_h.get(method_name)
    style_h2 = plot_styles_h2.get(method_name)

    plt.plot(xs_h, ys_h[:, 0], style_h['color'], linestyle=style_h['linestyle'], marker=style_h['marker'], 
             markersize=5, markevery=max(1, len(xs_h)//10), label=f"Численное решение ({method_name}, h={h})")
    
    plt.plot(xs_h2, ys_h2[:, 0], style_h2['color'], linestyle=style_h2['linestyle'], marker=style_h2['marker'], 
             markersize=5, markevery=max(1, len(xs_h2)//10), label=f"Численное решение ({method_name}, h={h2})")
    
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title(f"Сравнение истинного и численных решений методом {method_name}", fontsize=16)
    plt.xlabel("x", fontsize=12)
    plt.ylabel("y(x)", fontsize=12)
    plt.xlim(A_INTERVAL, B_INTERVAL)
    plt.tick_params(axis='both', which='major', labelsize=10)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    graph_filepath_individual = os.path.join(output_dir, f'{method_name.lower().replace(" ", "_")}_comparison.png')
    plt.savefig(graph_filepath_individual, dpi=300)
    print(f"График для метода {method_name} сохранен в файл {graph_filepath_individual}")


# ==================================================================================
# 6. Основная логика выполнения
# ==================================================================================

def main():
    """
    Главная функция для выполнения всех расчетов и анализа.
    """
    print("===================================================================")
    print("Начало выполнения программы")
    print("===================================================================")

    # 1. Валидация входных параметров
    try:
        validate_initial_parameters(A_INTERVAL, B_INTERVAL, Y0_INITIAL, H_STEP)
    except (TypeError, ValueError) as e:
        print(f"Ошибка инициализации: {e}")
        return

    # 2. Генерация сетки точек
    xs = generate_grid_points(A_INTERVAL, B_INTERVAL, H_STEP)
    if not xs:
        print("Ошибка: Список узлов xs пуст. Проверьте интервал и шаг.")
        return

    print(f"Генерация точек `xs` на интервале [{A_INTERVAL}, {B_INTERVAL}] с шагом h={H_STEP}.")
    print(f"Начальные условия `y0` заданы как {Y0_INITIAL}.")
    print(f"Для метода Адамса начальные 4 точки получаются из решения Рунге-Кутты с тем же шагом.")
    print("===================================================================")

    # 3. Решение ОДУ численными методами (шаг h)
    print(f"Решение ОДУ с шагом: {H_STEP}")
    ys_euler = solve_euler(xs, Y0_INITIAL, H_STEP, f)
    ys_rungekutta = solve_runge_kutta(xs, Y0_INITIAL, H_STEP, f, RK_P, RK_AS, RK_BS, RK_CS)
    
    if len(ys_rungekutta) < 4:
        print("Ошибка: Недостаточно точек Рунге-Кутты для инициализации метода Адамса.")
        return
    
    # Для Адамса нужны первые 4 точки из Рунге-Кутты
    ys_adams = solve_adams(xs, ys_rungekutta[:len(xs)], H_STEP, f) 

    # 4. Вывод результатов и сравнение с истинным решением (шаг h)
    print("\n===================================================================")
    print("Сравнение численных решений с истинным (шаг h):")
    print("===================================================================")
    for i in range(len(xs)):
        y_true = get_true_solution(xs[i])

        print(f"xk = {np.round(xs[i], 5)}, y(xk) = {np.round(y_true, 5)}")

        error_euler = abs(ys_euler[i][0] - y_true)
        error_rungekutta = abs(ys_rungekutta[i][0] - y_true)
        error_adams = abs(ys_adams[i][0] - y_true)

        print(f"\tЭйлер:      yk = {np.round(ys_euler[i][0], 5)}, e = {np.round(error_euler, 8)}")
        print(f"\tРунге-Кутт: yk = {np.round(ys_rungekutta[i][0], 5)}, e = {np.round(error_rungekutta, 8)}")
        print(f"\tАдамс:      yk = {np.round(ys_adams[i][0], 5)}, e = {np.round(error_adams, 8)}")

    # 5. Решение ОДУ численными методами (шаг h/2)
    H_STEP_HALF = H_STEP / 2
    if H_STEP_HALF <= 0:
        print("Ошибка: Шаг H_STEP_HALF (h/2) должен быть положительным числом.")
        return

    xs_half = generate_grid_points(A_INTERVAL, B_INTERVAL, H_STEP_HALF)
    if not xs_half:
        print("Ошибка: Список узлов xs_half пуст. Проверьте интервал и шаг.")
        return

    ys_euler_half = solve_euler(xs_half, Y0_INITIAL, H_STEP_HALF, f)
    ys_rungekutta_half = solve_runge_kutta(xs_half, Y0_INITIAL, H_STEP_HALF, f, RK_P, RK_AS, RK_BS, RK_CS)
    
    if len(ys_rungekutta_half) < 4:
        print("Ошибка: Недостаточно точек Рунге-Кутты (для h/2) для инициализации метода Адамса.")
        return

    ys_adams_half = solve_adams(xs_half, ys_rungekutta_half[:len(xs_half)], H_STEP_HALF, f) 

    # 6. Апостериорные оценки погрешности по Рунге
    print("\n===================================================================")
    print("Апостериорные оценки погрешности по Рунге:")
    print(f"\tЭйлер (порядок 1):      {calculate_runge_error(ys_euler, ys_euler_half, 1)}")
    print(f"\tРунге-Кутт (порядок 4): {calculate_runge_error(ys_rungekutta, ys_rungekutta_half, 4)}")
    print(f"\tАдамс (порядок 4):      {calculate_runge_error(ys_adams, ys_adams_half, 4)}")

    # 7. Построение графиков
    true_y_values = np.array([get_true_solution(x) for x in xs])
    methods_results_h = {
        "Эйлер": ys_euler,
        "Рунге-Кутта": ys_rungekutta,
        "Адамс": ys_adams
    }
    plot_results_combined(xs, true_y_values, methods_results_h, 
                          OUTPUT_DIRECTORY, 'comparison_ode_methods.png', 
                          f"Сравнение истинного и численных решений ОДУ (шаг h={H_STEP})")

    # Индивидуальные графики для сравнения h и h/2
    plot_results_individual(xs, ys_euler, xs_half, ys_euler_half, get_true_solution, 
                            OUTPUT_DIRECTORY, "Эйлер", H_STEP, H_STEP_HALF)
    plot_results_individual(xs, ys_rungekutta, xs_half, ys_rungekutta_half, get_true_solution, 
                            OUTPUT_DIRECTORY, "Рунге-Кутта", H_STEP, H_STEP_HALF)
    plot_results_individual(xs, ys_adams, xs_half, ys_adams_half, get_true_solution, 
                            OUTPUT_DIRECTORY, "Адамс", H_STEP, H_STEP_HALF)

    # 8. Расчет и вывод максимальных абсолютных ошибок
    print("\n===================================================================")
    print("Максимальные абсолютные ошибки:")

    max_error_euler_h = calculate_max_absolute_error(xs, ys_euler, get_true_solution)
    max_error_rungekutta_h = calculate_max_absolute_error(xs, ys_rungekutta, get_true_solution)
    max_error_adams_h = calculate_max_absolute_error(xs, ys_adams, get_true_solution)

    print(f"При шаге h = {H_STEP}:")
    print(f"\tМаксимальная абсолютная ошибка (Эйлер):      {max_error_euler_h:.8e}")
    print(f"\tМаксимальная абсолютная ошибка (Рунге-Кутт): {max_error_rungekutta_h:.8e}")
    print(f"\tМаксимальная абсолютная ошибка (Адамс):      {max_error_adams_h:.8e}")

    max_error_euler_h2 = calculate_max_absolute_error(xs_half, ys_euler_half, get_true_solution)
    max_error_rungekutta_h2 = calculate_max_absolute_error(xs_half, ys_rungekutta_half, get_true_solution)
    max_error_adams_h2 = calculate_max_absolute_error(xs_half, ys_adams_half, get_true_solution)

    print(f"При шаге h/2 = {H_STEP_HALF}:")
    print(f"\tМаксимальная абсолютная ошибка (Эйлер):      {max_error_euler_h2:.8e}")
    print(f"\tМаксимальная абсолютная ошибка (Рунге-Кутт): {max_error_rungekutta_h2:.8e}")
    print(f"\tМаксимальная абсолютная ошибка (Адамс):      {max_error_adams_h2:.8e}")

    # 9. Проверка на обусловленность задачи (Чувствительность к начальным условиям)
    print("\n===================================================================")
    print("ПРОВЕРКА НА ОБУСЛОВЛЕННОСТЬ ЗАДАЧИ (Чувствительность к начальным условиям):")

    x_test_start = A_INTERVAL
    eigenvalues_start = calculate_jacobian_eigenvalues(x_test_start)
    print(f"Собственные значения матрицы Якоби при x = {x_test_start:.2f}: {eigenvalues_start}")

    x_test_end = B_INTERVAL
    eigenvalues_end = calculate_jacobian_eigenvalues(x_test_end)
    print(f"Собственные значения матрицы Якоби при x = {x_test_end:.2f}: {eigenvalues_end}")

    is_ill_posed = False
    for val in eigenvalues_start:
        if np.real(val) > 0:
            is_ill_posed = True
            break

    print("\nАнализ жесткости:")
    print("    Для жестких систем требуется, чтобы действительные части всех собственных значений матрицы Якоби были ОТРИЦАТЕЛЬНЫМИ (Re λk < 0).")
    print(f"    В данном случае, собственные значения (для решения y' = Ay) равны 1 и 1/x (для x>0), и их действительные части всегда положительны на интервале интегрирования [{A_INTERVAL}, {B_INTERVAL}].")
    print("    Таким образом, условие Re λk < 0 не выполняется, и система НЕ является жесткой.")

    if is_ill_posed:
        print("--> Задача является чувствительной к начальным условиям (плохо обусловленной),")
        print("    так как существуют собственные значения матрицы Якоби с положительными вещественными частями.")
        print("    Это значит, что малые изменения начальных условий могут привести к экспоненциальному росту ошибок в решении.")
        print(f"    (Например, собственные значения при x={x_test_start:.2f} равны {eigenvalues_start.round(5)} и при x={x_test_end:.2f} равны {eigenvalues_end.round(5)}.)")
        print(f"    Их вещественные части положительны на интервале [{A_INTERVAL}, {B_INTERVAL}].")
        print(      "    Кроме того, свойство устойчивости как равномерной зависимости от h решения разностной задачи относительно возмущения правых частей и граничных условий не выполнено.")
        print("     Свойство аппроксимации проявляется в порядке точности каждого из рассмотренных методов.")
        print("     Схемы аппроксимирующие, так как Ψ→0 при h→0 в разложении в ряд Тейлора.")
        print("     Схемы Эйлера, Рунге-Кутты, Адамса являются явными. В них решение в последующей точке определяется непосредственно алгебраическими соотношениями с известными коэф.")
        print("     При расчете на шаге явные схемы требуют меньшего числа операций, чем неявные. При этом менее устойчивы, чем неявные. Ограничения на шаг жесткие.")
    else:
        print("--> Задача хорошо обусловлена, все собственные значения матрицы Якоби имеют неположительные вещественные части.")
        print("     Кроме того, свойство устойчивости выполнено.")
        print("     Свойство аппроксимации проявляется в порядке точности каждого из рассмотренных методов.")
        print("     Схемы аппроксимирующие, так как Ψ​→0 при h→0 в разложении в ряд Тейлора.")
        print("     Схемы Эйлера, Рунге-Кутты, Адамса являются явными. В них решение в последующей точке определяется непосредственно алгебраическими соотношениями с известными коэф.")
        print("     При расчете на шаге явные схемы требуют меньшего числа операций, чем неявные. При этом менее устойчивы, чем неявные. Ограничения на шаг жесткие.")


    print("===================================================================")

    print("===================================================================")
    print("ПОРЯДКИ ТОЧНОСТИ МЕТОДОВ:")
    print("-> Метод Эйлера:")
    print("   - 1-й порядок аппроксимации на интервале.")
    print("   - 1-й порядок точности относительно шага h: O(h).")
    print("-> Метод Рунге-Кутты:")
    print("   - Типичная схема (например, Рунге-Кутты 4-го порядка): O(h^4) (глобальный порядок точности).")
    print("-> Метод Адамса:")
    print("   - Порядок точности на шаге (локальная ошибка дискретизации): O(h^5).")
    print("   - Порядок точности на интервале (глобальная ошибка): O(h^4).")
    print("===================================================================")

    # 10. Проверка по теоретическим оценкам погрешности
    M4_val = get_true_solution_4th_derivative(B_INTERVAL)

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

    theoretical_error_adams_h = (B_INTERVAL - A_INTERVAL) / 3 * (H_STEP ** 4) * M4_val 
    theoretical_error_adams_h2 = (B_INTERVAL - A_INTERVAL) / 3 * (H_STEP_HALF ** 4) * M4_val

    print(f"\nТеоретическая оценка погрешности для Адамса (p=4) [используя формулу E ~ (b-a)/3 * h^4 * M4]:")
    print(f"\tПри шаге h = {H_STEP}:     E_theory = {theoretical_error_adams_h:.8e}")
    print(f"\tФактическая ошибка:   E_actual = {max_error_adams_h:.8e}")
    if max_error_adams_h <= theoretical_error_adams_h:
        print("\t-> Фактическая ошибка меньше или равна теоретической оценке (GOOD).")
    else:
        print("\t-> Фактическая ошибка превышает теоретическую оценку (CAUTION).")

    print(f"\tПри шаге h/2 = {H_STEP_HALF}: E_theory = {theoretical_error_adams_h2:.8e}")
    print(f"\tФактическая ошибка:   E_actual = {max_error_adams_h2:.8e}")
    if max_error_adams_h2 <= theoretical_error_adams_h2:
        print("\t-> Фактическая ошибка меньше или равна теоретической оценке (GOOD).")
    else:
        print("\t-> Фактическая ошибка превышает теоретическую оценку (CAUTION).")

    print("===================================================================")
    print("Завершение выполнения программы.")
    print("===================================================================")


if __name__ == "__main__":
    main()
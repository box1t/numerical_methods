import numpy as np
import matplotlib.pyplot as plt
import os
import sys

try:
    from odu import RK_P, RK_AS, RK_BS, RK_CS, solve_runge_kutta, calculate_runge_error, generate_grid_points
except ImportError:
    print("Ошибка: Не удалось импортировать функции из odu.py.")
    print("Пожалуйста, убедитесь, что odu.py находится в той же директории или доступен в PYTHONPATH.")
    sys.exit(1)

A_INTERVAL = 0
B_INTERVAL = 1
Y0_INITIAL_GUESS_FOR_RK = np.array([-2, 0])
EPSILON = 1e-6
H_STEP = 0.1

OUTPUT_DIRECTORY = '/home/snowwy/Desktop/MAI/_math/8_Численные_методы/numerical_methods/lab_4/src'

def f(x: float, y: np.ndarray) -> np.ndarray:
    if not isinstance(x, (int, float)):
        raise TypeError("x должен быть числом.")
    if not isinstance(y, np.ndarray) or y.ndim != 1 or y.shape[0] != 2:
        raise TypeError("y должен быть одномерным numpy.ndarray размерности 2.")

    if abs(x - 1) < 1e-9:
        return np.array([
            y[1],
            y[1] / 2
        ])
    
    denominator = x**2 - 1
    if abs(denominator) < 1e-10:
        raise ZeroDivisionError(f"Деление на ноль в f(x,y): x = {x}, x^2 - 1 = {denominator}. "
                                f"Убедитесь, что интервал интегрирования не включает x=1 или x=-1 без специальной обработки.")
    
    return np.array([
        y[1],
        (y[0] - (x - 3) * y[1]) / denominator
    ])

def get_true_solution(x: float) -> float:
    if not isinstance(x, (int, float)):
        raise TypeError("x должен быть числом.")
    if abs(x + 1) < 1e-9:
        raise ZeroDivisionError("Аналитическое решение не определено при x = -1.")
    return x - 3 + 1 / (x + 1)

def get_target_boundary_condition_value(ys: np.ndarray) -> float:
    if not isinstance(ys, np.ndarray) or ys.ndim != 2:
        raise TypeError("ys должен быть двумерным numpy.ndarray.")
    if ys.shape[1] != 2:
        raise ValueError("ys должен содержать y и y'.")
        
    if ys.shape[0] == 0:
        raise ValueError("Входной массив ys пуст.")
        
    return ys[-1][0] + ys[-1][1] + 0.75

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
    rk_cs: list,
    verbose: bool = True
) -> tuple[np.ndarray, int, float]:
    initial_y0_val = float(initial_y0_val)

    eta0 = 0.3
    eta1 = 0.5

    ys0 = solve_runge_kutta(x_points, np.array([initial_y0_val, eta0]), h_step, ode_func, rk_p, rk_as, rk_bs, rk_cs)
    ys1 = solve_runge_kutta(x_points, np.array([initial_y0_val, eta1]), h_step, ode_func, rk_p, rk_as, rk_bs, rk_cs)

    F0 = target_bc_func(ys0)
    F1 = target_bc_func(ys1)

    iterations = 0
    if verbose:
        print("\n--- Процесс поиска корня Ф(\u03B7) = 0 (Метод секущих) ---")
        print(f"Итерация {iterations}: \u03B7 = {eta0:.6e}, Ф(\u03B7) = {F0:.6e}")
    
    iterations += 1
    if verbose:
        print(f"Итерация {iterations}: \u03B7 = {eta1:.6e}, Ф(\u03B7) = {F1:.6e}")

    if abs(F1) < tolerance:
        if verbose:
            print(f"Сходимость достигнута на итерации {iterations} (первое приближение) с Ф(\u03B7) = {F1:.6e}")
        return ys1, iterations, eta1

    while True:
        if abs(F1 - F0) < 1e-12:
            if abs(F1) < tolerance:
                if verbose:
                    print(f"Сходимость достигнута на итерации {iterations} с Ф(\u03B7) = {F1:.6e} (погрешность F1-F0 очень мала).")
                return ys1, iterations, eta1
            else:
                print(f"Предупреждение: Разница Ф(\u03B71) - Ф(\u03B70) ({F1-F0:.2e}) очень мала, но Ф(\u03B7) еще не сошлось. Метод секущих может застрять.")
                if iterations > 100: 
                    raise RuntimeError(f"Метод стрельбы не сошелся из-за слишком малой разницы Ф(\u03B71) - Ф(\u03B70) на итерации {iterations}.")
                else:
                    eta1 += tolerance * 10 
                    ys1 = solve_runge_kutta(x_points, np.array([initial_y0_val, eta1]), h_step, ode_func, rk_p, rk_as, rk_bs, rk_cs)
                    F1 = target_bc_func(ys1)
                    if verbose:
                        print(f"Итерация {iterations}: Восстановление - новое \u03B71 = {eta1:.6e}, Ф(\u03B71) = {F1:.6e}")
                    continue 

        eta = eta1 - F1 * (eta1 - eta0) / (F1 - F0)

        ys = solve_runge_kutta(x_points, np.array([initial_y0_val, eta]), h_step, ode_func, rk_p, rk_as, rk_bs, rk_cs)
        
        F0 = F1
        F1 = target_bc_func(ys)
        eta0 = eta1
        eta1 = eta

        iterations += 1
        if verbose:
            print(f"Итерация {iterations}: \u03B7 = {eta:.6e}, Ф(\u03B7) = {F1:.6e}")

        if abs(F1) < tolerance:
            if verbose:
                print(f"Сходимость достигнута на итерации {iterations} с Ф(\u03B7) = {F1:.6e}")
            return ys, iterations, eta
        
        if iterations > 1000:
            print(f"Предупреждение: Превышено максимальное количество итераций (1000). Метод стрельбы не сошелся. Ф(\u03B7) = {F1:.6e}")
            return ys, iterations, eta

def plot_bvp_solution(xs: list, true_y_values: np.ndarray, numerical_solution: np.ndarray, 
                      output_dir: str, filename: str, title: str):
    plt.figure(figsize=(12, 7))

    plt.plot(xs, true_y_values, 'k-', linewidth=2, label="Истинное решение")
    plt.plot(xs, numerical_solution[:, 0], 'r--', linewidth=1.5, marker='o', markersize=4, 
             markevery=max(1, len(xs)//10), label="Численное решение (Метод Стрельбы)")

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
    print(f"\nГрафик сохранен в файл: {graph_filepath}")

if __name__ == "__main__":
    print("===================================================================")
    print("Начало выполнения программы для краевой задачи методом стрельбы")
    print("===================================================================")
    print(f"Интервал: [{A_INTERVAL}, {B_INTERVAL}]")
    print(f"Шаг интегрирования: {H_STEP}")
    print(f"Точность для метода стрельбы (epsilon): {EPSILON}")

    xs = generate_grid_points(A_INTERVAL, B_INTERVAL, H_STEP)
    if not xs:
        print("Ошибка: Список узлов xs пуст. Проверьте интервал и шаг.")
        sys.exit(1)

    ys_rungekutta_ivp = solve_runge_kutta(xs, Y0_INITIAL_GUESS_FOR_RK, H_STEP, f, RK_P, RK_AS, RK_BS, RK_CS)

    print(f"\nРешение краевой задачи методом стрельбы с начальным условием y({A_INTERVAL})={Y0_INITIAL_GUESS_FOR_RK[0]}:")
    ys_shooting, iter_shooting, optimal_eta = solve_shooting_method(
        xs, Y0_INITIAL_GUESS_FOR_RK[0], H_STEP, EPSILON, f, get_target_boundary_condition_value,
        RK_P, RK_AS, RK_BS, RK_CS, verbose=True
    )
    print(f"Общее число итераций в стрельбе: {iter_shooting}")
    print(f"Найденное оптимальное y'({A_INTERVAL}) (\u03B7) для краевого условия: {optimal_eta:.8e}")
    
    if ys_shooting is not None and len(ys_shooting) > 0:
        y_at_b = ys_shooting[-1][0]
        yp_at_b = ys_shooting[-1][1]
        print(f"-------------------------------------------------------------------")
        print(f"Найденное решение задачи Коши (после сходимости метода стрельбы) на правом конце отрезка x = {B_INTERVAL}:")
        print(f"y({B_INTERVAL}) = {y_at_b:.8e}")
        print(f"y'({B_INTERVAL}) = {yp_at_b:.8e}")
        print(f"Проверка краевого условия y({B_INTERVAL}) + y'({B_INTERVAL}) + 0.75 = {y_at_b + yp_at_b + 0.75:.8e} (должно быть близко к 0)")
        print(f"-------------------------------------------------------------------")

    print("\n===================================================================")
    print("Сравнение численных решений с истинным (шаг h):")
    print("===================================================================")
    
    for i in range(len(xs)):
        try:
            y_true = get_true_solution(xs[i])
        except ZeroDivisionError as e:
            print(f"Ошибка при вычислении истинного решения в x={xs[i]}: {e}. Пропуск точки.")
            continue

        error_rungekutta_ivp = abs(ys_rungekutta_ivp[i][0] - y_true)
        error_shooting = abs(ys_shooting[i][0] - y_true)

        print(f"xk = {xs[i]:.5g}, y_true(xk) = {y_true:.5g}")
        print(f"        Рунге-Кутт: yk = {ys_rungekutta_ivp[i][0]:.5g}, e = {error_rungekutta_ivp:.16f}")
        print(f"        Стрельба:          yk = {ys_shooting[i][0]:.5g}, e = {error_shooting:.16f}")

    H_STEP_HALF = H_STEP / 2
    xs_half = generate_grid_points(A_INTERVAL, B_INTERVAL, H_STEP_HALF)
    if not xs_half:
        print("Ошибка: Список узлов xs_half пуст. Проверьте интервал и шаг.")
        sys.exit(1)

    print(f"\nПовторное решение методом стрельбы для шага h/2 = {H_STEP_HALF} (для оценки погрешности по Рунге):")
    
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    ys_shooting_half, iter_shooting_half, _ = solve_shooting_method(
        xs_half, Y0_INITIAL_GUESS_FOR_RK[0], H_STEP_HALF, EPSILON, f, get_target_boundary_condition_value,
        RK_P, RK_AS, RK_BS, RK_CS, verbose=True
    )
    sys.stdout = original_stdout
    print(f"Итераций для h/2: {iter_shooting_half}")

    print("\n===================================================================")
    print("Апостериорная оценка погрешности по Рунге (для метода стрельбы):")
    runge_error = calculate_runge_error(ys_shooting[:, 0, np.newaxis], ys_shooting_half[:, 0, np.newaxis], RK_P)
    print(f"\tОценка погрешности для метода стрельбы (порядок {RK_P}): {runge_error:.8e}")
    print("===================================================================")

    true_y_values_for_plot = np.array([get_true_solution(x) for x in xs])
    
    methods_for_shooting_only_plot = {
        "Стрельба": ys_shooting
    }
    plot_bvp_solution(xs, true_y_values_for_plot, ys_shooting,
                          OUTPUT_DIRECTORY, '4.2_shooting_.png',
                          f"Решение краевой задачи методом стрельбы (h={H_STEP})")
    
    print("\n===================================================================")
    print("Завершение выполнения программы.")
    print("===================================================================")
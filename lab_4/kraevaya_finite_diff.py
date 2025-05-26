import numpy as np
import os
import matplotlib.pyplot as plt
from ..lab_1.progonka_lab import progonka

def p(x):
    denominator = x**2 - 1
    if abs(denominator) < 1e-12:
        raise ZeroDivisionError(f"Деление на ноль в p(x) при x = {x:.5f} (x^2-1 = {denominator:.2e}).")
    return (x - 3) / denominator

def q(x):
    denominator = x**2 - 1
    if abs(denominator) < 1e-12:
        raise ZeroDivisionError(f"Деление на ноль в q(x) при x = {x:.5f} (x^2-1 = {denominator:.2e}).")
    return -1 / denominator

def f(x):
    return 0

def get_true_solution(x):
    return x - 3 + 1 / (x + 1)

def get_true_solution_4th_derivative(x):
    if abs(x + 1) < 1e-9:
        raise ZeroDivisionError(f"Сингулярность 4-й производной при x = {x:.5f}.")
    return 24 / ((x + 1)**5)

A_INTERVAL = 0
B_INTERVAL = 1 - 1e-9
#INITIAL_STEP_SIZE = 2**(-5)
INITIAL_STEP_SIZE = 0.1


def generate_grid_points(start_point, end_point, step_size):
    if not isinstance(start_point, (int, float)) or not isinstance(end_point, (int, float)):
        raise TypeError("Начальная и конечная точки интервала должны быть числами.")
    if start_point >= end_point:
        raise ValueError("Начальная точка интервала должна быть меньше конечной.")
    if not isinstance(step_size, (int, float)) or step_size <= 0:
        raise ValueError("Шаг h должен быть положительным числом.")

    grid_points = []
    current_x = float(start_point)
    while current_x <= end_point + 1e-10:
        grid_points.append(current_x)
        current_x += step_size
        if len(grid_points) > 100000:
            raise RuntimeError("Слишком много точек сетки. Возможно, шаг h слишком мал или интервал слишком большой.")
    
    if not grid_points or abs(grid_points[-1] - end_point) > 1e-10 :
        grid_points.append(float(end_point))

    grid_points = sorted(list(set(grid_points)))
    
    return grid_points


def solve_finite_difference(num_points, grid_points, step_size, A_b1, A_c1, A_an, A_bn, b1, bn):
    if not isinstance(num_points, int) or num_points <= 1:
        raise ValueError("Количество точек (num_points) должно быть целым числом больше 1.")
    if not isinstance(grid_points, list) or not all(isinstance(x, (int, float)) for x in grid_points):
        raise TypeError("grid_points должен быть списком чисел.")
    if len(grid_points) != num_points:
        pass

    if not all(isinstance(arg, (int, float)) for arg in [A_b1, A_c1, A_an, A_bn, b1, bn]):
        raise TypeError("Коэффициенты и значения граничных условий должны быть числами.")

    matrix_A = np.zeros((num_points, 3))
    vector_b = np.empty(num_points)

    matrix_A[0][1] = A_b1
    matrix_A[0][2] = A_c1
    vector_b[0] = b1

    matrix_A[num_points - 1][0] = A_an
    matrix_A[num_points - 1][1] = A_bn
    vector_b[num_points - 1] = bn

    for k in range(1, num_points - 1):
        current_x = grid_points[k]
        try:
            pk = p(current_x)
            qk = q(current_x)
            fk = f(current_x)
        except ZeroDivisionError as e:
            raise ValueError(f"Деление на ноль в коэффициентах p(x) или q(x) при x = {current_x:.5f}. Это указывает на сингулярность ОДУ в этой точке: {e}")

        matrix_A[k][0] = 1 - pk * step_size / 2
        matrix_A[k][1] = -2 + step_size ** 2 * qk
        matrix_A[k][2] = 1 + pk * step_size / 2
        vector_b[k] = step_size ** 2 * fk

    solutions_y = progonka(matrix_A, vector_b)
    return solutions_y

def calculate_runge_error(fine_solution: np.ndarray, coarse_solution: np.ndarray, order_of_accuracy):
    if not isinstance(fine_solution, np.ndarray) or not isinstance(coarse_solution, np.ndarray):
        raise TypeError("Решения должны быть массивами NumPy.")
    if not isinstance(order_of_accuracy, (int, float)) or order_of_accuracy <= 0:
        raise ValueError("Порядок точности (order_of_accuracy) должен быть положительным числом.")

    k = 2

    if not (fine_solution.shape[0] >= (coarse_solution.shape[0] -1) * k + 1):
        raise ValueError(f"Длины массивов решений несовместимы для оценки Рунге. "
                         f"fine_solution.shape[0]={fine_solution.shape[0]}, "
                         f"coarse_solution.shape[0]={coarse_solution.shape[0]}, k={k}. "
                         f"Ожидается fine_solution.shape[0] >= (coarse_solution.shape[0]-1)*k + 1")

    error_max = 0.0
    denominator = (k ** order_of_accuracy - 1)
    
    if abs(denominator) < 1e-10:
        raise ZeroDivisionError("Знаменатель в формуле Рунге слишком мал. Проверьте порядок точности (order_of_accuracy).")

    for i in range(coarse_solution.shape[0]):
        error_current = abs(fine_solution[i * k] - coarse_solution[i]) / denominator
        error_max = max(error_max, error_current)
        
    return error_max

def calculate_jacobian_eigenvalues(x_value):
    if not isinstance(x_value, (int, float)):
        raise TypeError("x_value должен быть числом.")
    
    try:
        px = p(x_value)
        qx = q(x_value)
    except ZeroDivisionError as e:
        raise ZeroDivisionError(f"Деление на ноль при вычислении коэффициентов p(x) или q(x) для матрицы Якоби при x = {x_value:.5f}. Это указывает на сингулярность исходного ОДУ: {e}")

    discriminant = px**2 - 4 * qx
    
    if discriminant < 0:
        lambda1 = (-px + np.sqrt(complex(discriminant))) / 2
        lambda2 = (-px - np.sqrt(complex(discriminant))) / 2
    else:
        lambda1 = (-px + np.sqrt(discriminant)) / 2
        lambda2 = (-px - np.sqrt(discriminant)) / 2
    
    return np.array([lambda1, lambda2])


def perform_condition_and_stiffness_analysis(start_interval, end_interval):
    if not isinstance(start_interval, (int, float)) or not isinstance(end_interval, (int, float)):
        raise TypeError("Границы интервала должны быть числами.")
    if start_interval >= end_interval:
        raise ValueError("Начало интервала должно быть меньше конца интервала для анализа.")

    print("\nПРОВЕРКА НА ОБУСЛОВЛЕННОСТЬ ЗАДАЧИ (Чувствительность к начальным условиям):")
    print("Анализ жесткости:")
    print("    Для жестких систем требуется, чтобы действительные части всех собственных значений матрицы Якоби были ОТРИЦАТЕЛЬНЫМИ (Re λk < 0) и существенно отличались по модулю.")

    try:
        eigenvalues_at_start = calculate_jacobian_eigenvalues(start_interval)
        print(f"Собственные значения матрицы Якоби при x = {start_interval:.2f}: {np.round(eigenvalues_at_start, 5)}")
    except ZeroDivisionError as e:
        print(f"Ошибка при вычислении собственных значений в начале интервала (x={start_interval:.2f}): {e}")
        print("    Невозможно выполнить полную проверку обусловленности/жесткости из-за сингулярности.")
        eigenvalues_at_start = np.array([np.nan, np.nan])


    try:
        eigenvalues_at_end = calculate_jacobian_eigenvalues(end_interval)
        print(f"Собственные значения матрицы Якоби при x = {end_interval:.2f}: {np.round(eigenvalues_at_end, 5)}")
    except ZeroDivisionError as e:
        print(f"Ошибка при вычислении собственных значений в конце интервала (x={end_interval:.2f}): {e}")
        print("    Невозможно выполнить полную проверку обусловленности/жесткости из-за сингулярности.")
        eigenvalues_at_end = np.array([np.nan, np.nan])


    is_ill_posed_or_problematic = False
    all_real_parts_non_positive = True
    
    num_check_points = 10
    
    check_points = np.linspace(start_interval, end_interval, num_check_points)

    for x_val in check_points:
        try:
            eigs = calculate_jacobian_eigenvalues(x_val)
            for val in eigs:
                if np.real(val) > 1e-9:
                    is_ill_posed_or_problematic = True
                    all_real_parts_non_positive = False
                    break
            if not all_real_parts_non_positive:
                break
        except ZeroDivisionError:
            print(f"    Предупреждение: сингулярность в коэффициентах p(x) или q(x) при x={x_val:.2f}. Это может указывать на плохую обусловленность или жесткость.")
            is_ill_posed_or_problematic = True
            all_real_parts_non_positive = False
        except Exception as e:
            print(f"    Ошибка при вычислении собственных значений при x={x_val:.2f}: {e}")
            is_ill_posed_or_problematic = True
            all_real_parts_non_positive = False

    if all_real_parts_non_positive:
        print("    На интервале интегрирования, все вещественные части собственных значений матрицы Якоби являются неположительными (Re λk <= 0).")
        print("    Поэтому, система НЕ является жесткой.")
    else:
        example_eigs_str = ""
        if not np.isnan(eigenvalues_at_end).any():
            example_eigs_str = f" (например, при x={end_interval:.2f}, собственные значения: {eigenvalues_at_end[0]:.5f}, {eigenvalues_at_end[1]:.5f})."
        elif not np.isnan(eigenvalues_at_start).any():
             example_eigs_str = f" (например, при x={start_interval:.2f}, собственные значения: {eigenvalues_at_start[0]:.5f}, {eigenvalues_at_start[1]:.5f})."

        print(f"    На интервале интегрирования существуют собственные значения матрицы Якоби с положительными вещественными частями{example_eigs_str}")
        print("    Поэтому, система НЕ является жесткой (т.к. нет строго отрицательных вещественных частей для всех λk).")

    if is_ill_posed_or_problematic:
        print("    Задача потенциально плохо обусловлена или имеет сложности, такие как сингулярности в коэффициентах или положительные вещественные части собственных значений.")
    else:
        print("    Задача, судя по собственным значениям Якоби, является хорошо обусловленной на интервале.")


def plot_solution(x_values, numerical_y_values, true_y_values, output_directory, filename="odu_4_3_finite_diff.png"):
    if not isinstance(x_values, list) or not all(isinstance(x, (int, float)) for x in x_values):
        raise TypeError("x_values должен быть списком чисел.")
    if not isinstance(numerical_y_values, np.ndarray) or not isinstance(true_y_values, np.ndarray):
        raise TypeError("numerical_y_values и true_y_values должны быть массивами NumPy.")
    if not isinstance(output_directory, str) or not output_directory:
        raise ValueError("output_directory должен быть непустой строкой.")
    if not isinstance(filename, str) or not filename:
        raise ValueError("filename должен быть непустой строкой.")

    os.makedirs(output_directory, exist_ok=True)
    graph_filepath = os.path.join(output_directory, filename)

    plt.figure(figsize=(14, 8))

    plt.plot(x_values, true_y_values, 'k-', linewidth=2, label="Истинное решение")
    plt.plot(x_values, numerical_y_values, 'b--', marker='o', markersize=5, markevery=max(1, len(x_values)//10), label="Метод конечных разностей")

    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title("Сравнение истинного и численного решений ОДУ (Метод конечных разностей)", fontsize=16)
    plt.xlabel("x", fontsize=12)
    plt.ylabel("y(x)", fontsize=12)
    plt.xlim(A_INTERVAL, B_INTERVAL)
    plt.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()
    plt.savefig(graph_filepath, dpi=300)
    print(f"\nГрафик сохранен в файл {graph_filepath}")


def main():
    print("Начало решения краевой задачи методом конечных разностей.")

    if not isinstance(A_INTERVAL, (int, float)) or not isinstance(B_INTERVAL, (int, float)):
        raise TypeError("A_INTERVAL и B_INTERVAL должны быть числами.")
    if A_INTERVAL >= B_INTERVAL:
        raise ValueError("A_INTERVAL должен быть меньше B_INTERVAL.")
    if not isinstance(INITIAL_STEP_SIZE, (int, float)) or INITIAL_STEP_SIZE <= 0:
        raise ValueError("INITIAL_STEP_SIZE должен быть положительным числом.")

    print(f"Интервал интегрирования: [{A_INTERVAL}, {B_INTERVAL}]")
    print(f"Начальный шаг (h): {INITIAL_STEP_SIZE}\n")

    print("\n===================================================================")
    print("ВЫВОД УРАВНЕНИЙ, ИСПОЛЬЗУЕМЫХ В МЕТОДЕ КОНЕЧНЫХ РАЗНОСТЕЙ:")
    print("Основная краевая задача для ОДУ второго порядка имеет вид:")
    print("  u''(x) + p(x)u'(x) + q(x)u(x) = f(x)")
    print("  где: p(x) = (x-3)/(x^2-1), q(x) = -1/(x^2-1), f(x) = 0")
    print("\nЧисленное решение ищется на дискретной сетке x_k = A + k*h.")
    print("Производные аппроксимируются центральными разностями:")
    print("  u'(x_k)  ≈ (u_{k+1} - u_{k-1}) / (2h)  (порядок O(h^2))")
    print("  u''(x_k) ≈ (u_{k-1} - 2u_k + u_{k+1}) / (h^2) (порядок O(h^2))")
    print("\n1. Уравнения для внутренних узлов сетки (k = 1, ..., N-2):")
    print("   Подстановка аппроксимаций производных в исходное ОДУ приводит к:")
    print("     (1 - p(x_k)*h/2) * u_{k-1} + (-2 + h^2*q(x_k)) * u_k + (1 + p(x_k)*h/2) * u_{k+1} = h^2*f(x_k)")
    print("   Эти уравнения формируют основные строки трехдиагональной матрицы.")

    print("\n2. Уравнения, полученные из граничных условий (недостающие 2 уравнения):")
    print("   Для сохранения общего порядка точности O(h^2), граничные условия также аппроксимируются с порядком O(h^2).")
    print("   a) Левое граничное условие (при x = A_INTERVAL = 0): y'(0) = 0")
    print("      Используется фиктивная точка u_{-1} и ОДУ в точке x_0:")
    print("      u''(x_0) + p(x_0)u'(x_0) + q(x_0)u(x_0) = f(x_0)")
    print("      где u'(x_0) ≈ (u_1 - u_{-1}) / (2h) = 0  => u_{-1} = u_1")
    print("      Подстановка u_{-1} = u_1 в центральную разность для u''(x_0) дает:")
    print("      (-2/h^2 + q(x_0)) * u_0 + (2/h^2) * u_1 = f(x_0)")
    print("      Умножая на h^2: (-2 + h^2 * q(x_0)) * u_0 + 2 * u_1 = h^2 * f(x_0)")
    print("      Эта строка используется для первой строки в системе.")
    print("\n   b) Правое граничное условие (при x = B_INTERVAL = 1): y'(1) + y(1) = -0.75")
    print("      Используется фиктивная точка u_{N+1} и ОДУ в точке x_N:")
    print("      y'(x_N) ≈ (u_{N+1} - u_{N-1}) / (2h)")
    print("      Из граничного условия: (u_{N+1} - u_{N-1}) / (2h) + u_N = -0.75")
    print("      Выражаем u_{N+1} и подставляем в аппроксимацию ОДУ в точке x_N:")
    print("      2 * u_{N-1} + (-2 - 2h - h^2 * p(x_N) + h^2 * q(x_N)) * u_N = h^2 * f(x_N) + 1.5h + 0.75 * h^2 * p(x_N)")
    print("      Эта строка используется для последней строки в системе.")
    print("===================================================================\n")

    print("===================================================================")
    print("ПОРЯДКИ ТОЧНОСТИ МЕТОДОВ:")
    print("-> Метод конечных разностей:")
    print("   - При использовании центральных разностей и аппроксимаций граничных условий порядка O(h^2),")
    print("     глобальный порядок точности метода составляет O(h^2).")
    print("     Это означает, что при уменьшении шага h в 2 раза, ошибка должна уменьшаться примерно в 4 раза.")
    print("===================================================================")

    xs_h = generate_grid_points(A_INTERVAL, B_INTERVAL, INITIAL_STEP_SIZE)
    num_points_h = len(xs_h)
    print(f"Сетка из {num_points_h} точек с шагом h = {INITIAL_STEP_SIZE}.")

    bc1_coeff_y0 = -2 + INITIAL_STEP_SIZE**2 * q(A_INTERVAL)
    bc1_coeff_y1 = 2
    bc1_rhs = INITIAL_STEP_SIZE**2 * f(A_INTERVAL)

    try:
        p_B = p(B_INTERVAL)
        q_B = q(B_INTERVAL)
    except ZeroDivisionError as e:
        print(f"Предупреждение: Проблема с коэффициентами p(x) или q(x) на правой границе {B_INTERVAL}: {e}")
        p_B = p(B_INTERVAL)
        q_B = q(B_INTERVAL)


    bc2_coeff_y_N_minus_1 = 2
    bc2_coeff_y_N = -2 - 2*INITIAL_STEP_SIZE - INITIAL_STEP_SIZE**2 * p_B + INITIAL_STEP_SIZE**2 * q_B
    bc2_rhs = INITIAL_STEP_SIZE**2 * f(B_INTERVAL) + 1.5*INITIAL_STEP_SIZE + 0.75 * INITIAL_STEP_SIZE**2 * p_B


    solution_y_h = solve_finite_difference(num_points_h, xs_h, INITIAL_STEP_SIZE,
                                            A_b1=bc1_coeff_y0, A_c1=bc1_coeff_y1, b1=bc1_rhs,
                                            A_an=bc2_coeff_y_N_minus_1, A_bn=bc2_coeff_y_N, bn=bc2_rhs)

    print("\n--- Результаты численного решения и сравнение с истинным решением (шаг h) ---")
    total_error_h = 0.0
    for i in range(len(xs_h)):
        true_y_val = get_true_solution(xs_h[i])
        numerical_y_val = solution_y_h[i]
        absolute_error = abs(numerical_y_val - true_y_val)
        total_error_h = max(total_error_h, absolute_error) 
        print(f"x = {xs_h[i]:.5f}, Численное y(x) = {numerical_y_val:.10f}, Истинное y(x) = {true_y_val:.10f}, Абсолютная ошибка = {absolute_error:.16f}")
    print(f"\nМаксимальная абсолютная ошибка для шага h = {INITIAL_STEP_SIZE}: {total_error_h:.16f}")

    print("\n===================================================================")
    print("Расчет для шага в два раза короче (h/2) для оценки погрешности по Рунге:")
    half_step_size = INITIAL_STEP_SIZE / 2
    xs_h2 = generate_grid_points(A_INTERVAL, B_INTERVAL, half_step_size)
    num_points_h2 = len(xs_h2)
    print(f"Сетка из {num_points_h2} точек с шагом h/2 = {half_step_size}.")

    bc1_coeff_y0_h2 = -2 + half_step_size**2 * q(A_INTERVAL)
    bc1_coeff_y1_h2 = 2
    
    try:
        p_B_h2 = p(B_INTERVAL)
        q_B_h2 = q(B_INTERVAL)
    except ZeroDivisionError as e:
        print(f"Предупреждение: Проблема с коэффициентами p(x) или q(x) на правой границе {B_INTERVAL} для h/2: {e}")
        p_B_h2 = p(B_INTERVAL)
        q_B_h2 = q(B_INTERVAL)

    bc2_coeff_y_N_minus_1_h2 = 2
    bc2_coeff_y_N_h2 = -2 - 2*half_step_size - half_step_size**2 * p_B_h2 + half_step_size**2 * q_B_h2
    bc2_rhs_h2 = half_step_size**2 * f(B_INTERVAL) + 1.5*half_step_size + 0.75 * half_step_size**2 * p_B_h2


    solution_y_h2 = solve_finite_difference(num_points_h2, xs_h2, half_step_size,
                                            A_b1=bc1_coeff_y0_h2, A_c1=bc1_coeff_y1_h2, b1=bc1_rhs,
                                            A_an=bc2_coeff_y_N_minus_1_h2, A_bn=bc2_coeff_y_N_h2, bn=bc2_rhs_h2)

    max_error_h2 = 0.0
    for i in range(len(xs_h2)):
        true_y_val_h2 = get_true_solution(xs_h2[i])
        numerical_y_val_h2 = solution_y_h2[i]
        absolute_error_h2 = abs(numerical_y_val_h2 - true_y_val_h2)
        max_error_h2 = max(max_error_h2, absolute_error_h2)
    print(f"\nМаксимальная абсолютная ошибка для шага h/2 = {half_step_size}: {max_error_h2:.16f}")


    runge_error_value = calculate_runge_error(solution_y_h2, solution_y_h, 2)
    print(f"Апостериорная оценка погрешности по Рунге (порядок O(h^2)): {runge_error_value:.16f}")
    if max_error_h2 != 0:
        print(f"Отношение ошибок E_h / E_{{h/2}} = {total_error_h / max_error_h2:.4f} (ожидается ~4 для O(h^2) метода).")

    # perform_condition_and_stiffness_analysis(A_INTERVAL, B_INTERVAL)

    true_y_values_for_plot = np.array([get_true_solution(x) for x in xs_h])
    
    output_dir = '/home/snowwy/Desktop/MAI/_math/8_Численные_методы/numerical_methods/lab_4/src'
    plot_solution(xs_h, solution_y_h, true_y_values_for_plot, output_dir)

    print("\nЗавершение выполнения программы.")


if __name__ == "__main__":
    main()
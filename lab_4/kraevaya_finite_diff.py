import numpy as np
import os
import matplotlib.pyplot as plt
from ..lab_1.progonka_lab import progonka

def p(x):
    return (x - 3) / (x ** 2 - 1)

def q(x):
    return -1 / (x ** 2 - 1)

def f(x):
    return 0

def get_true_solution(x):
    return x - 3 + 1 / (x + 1)

def get_true_solution_4th_derivative(x):
    if abs(x + 1) < 1e-9:
        # y(x) = x - 3 + 1 / (x + 1)
        # y'(x) = 1 - 1 / (x + 1)^2
        # y''(x) = 2 / (x + 1)^3
        # y'''(x) = -6 / (x + 1)^4
        # y''''(x) = 24 / (x + 1)^5
        raise ZeroDivisionError(f"Зануление 4-й производной при x = {x:.5f}.")
    return 24 / ((x + 1)**5)

A_INTERVAL = 0
B_INTERVAL = 1
INITIAL_STEP_SIZE = 2**(-5)

def generate_grid_points(start_point, end_point, step_size):
    if not isinstance(start_point, (int, float)) or not isinstance(end_point, (int, float)):
        raise TypeError("Начальная и конечная точки интервала должны быть числами.")
    if start_point >= end_point:
        raise ValueError("Начальная точка интервала должна быть меньше конечной.")
    if not isinstance(step_size, (int, float)) or step_size <= 0:
        raise ValueError("Шаг h должен быть положительным числом.")

    grid_points = []
    current_x = float(start_point)
    while current_x < end_point + 1e-9:
        grid_points.append(current_x)
        current_x += step_size
        if len(grid_points) > 100000:
            raise RuntimeError("Слишком много точек сетки. Возможно, шаг h слишком мал или интервал слишком большой.")
    
    if not grid_points or abs(grid_points[-1] - end_point) > 1e-9 :
        if end_point not in grid_points:
            grid_points.append(float(end_point))

    grid_points = sorted(list(set(grid_points)))
    
    return grid_points


def solve_finite_difference(num_points, grid_points, step_size, A_b1, A_c1, A_an, A_bn, b1, bn):
    if not isinstance(num_points, int) or num_points <= 1:
        raise ValueError("Количество точек (num_points) должно быть целым числом больше 1.")
    if not isinstance(grid_points, list) or not all(isinstance(x, (int, float)) for x in grid_points):
        raise TypeError("grid_points должен быть списком чисел.")
    if len(grid_points) != num_points:
        print(f"Предупреждение: количество точек в сетке ({len(grid_points)}) не совпадает с num_points ({num_points}).")
    if not isinstance(step_size, (int, float)) or step_size <= 0:
        raise ValueError("Шаг h должен быть положительным числом.")
    
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
        except ZeroDivisionError:
            raise ValueError(f"Деление на ноль в коэффициентах p(x) или q(x) при x = {current_x:.5f}. Это указывает на сингулярность ОДУ в этой точке.")

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

    if coarse_solution.shape[0] * k - (k - 1) > fine_solution.shape[0] + 1:
        if fine_solution.shape[0] < coarse_solution.shape[0] * k - (k -1):
            raise ValueError("Длины массивов решений несовместимы для оценки Рунге. Убедитесь, что шаг h2 = h / k.")

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
    except ZeroDivisionError:
        raise ZeroDivisionError(f"Деление на ноль при вычислении коэффициентов p(x) или q(x) для матрицы Якоби при x = {x_value:.5f}. Это указывает на сингулярность исходного ОДУ.")

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
        return

    try:
        eigenvalues_at_end = calculate_jacobian_eigenvalues(end_interval)
        print(f"Собственные значения матрицы Якоби при x = {end_interval:.2f}: {np.round(eigenvalues_at_end, 5)}")
    except ZeroDivisionError as e:
        print(f"Ошибка при вычислении собственных значений в конце интервала (x={end_interval:.2f}): {e}")
        print("    Невозможно выполнить полную проверку обусловленности/жесткости из-за сингулярности.")
        return


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
            break
        except Exception as e:
            print(f"    Ошибка при вычислении собственных значений при x={x_val:.2f}: {e}")
            is_ill_posed_or_problematic = True
            all_real_parts_non_positive = False
            break

    if all_real_parts_non_positive:
        print("    На интервале интегрирования, все вещественные части собственных значений матрицы Якоби являются неположительными (Re λk <= 0).")
        print("    Поэтому, система НЕ является жесткой.")
    else:
        example_eigs_str = ""
        if 'eigenvalues_at_end' in locals() and not np.isnan(eigenvalues_at_end).any():
            example_eigs_str = f" (например, при x={end_interval:.2f}, собственные значения: {eigenvalues_at_end[0]:.5f}, {eigenvalues_at_end[1]:.5f})."
        else:
            if 'eigenvalues_at_start' in locals() and not np.isnan(eigenvalues_at_start).any():
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
    print("  u'(x_k)  ≈ (u_{k+1} - u_{k-1}) / (2h)")
    print("  u''(x_k) ≈ (u_{k-1} - 2u_k + u_{k+1}) / (h^2)")
    print("\n1. Уравнения для внутренних узлов сетки (k = 1, ..., N-2):")
    print("   Подстановка аппроксимаций производных в исходное ОДУ приводит к:")
    print("     (1 - p(x_k)*h/2) * u_{k-1} + (-2 + h^2*q(x_k)) * u_k + (1 + p(x_k)*h/2) * u_{k+1} = h^2*f(x_k)")
    print("   Эти уравнения формируют основные строки трехдиагональной матрицы.")

    print("\n2. Уравнения, полученные из граничных условий (недостающие 2 уравнения):")
    print("   a) Левое граничное условие (при x = A_INTERVAL = 0): y'(0) = 0")
    print("      Аппроксимируется правой разностью: (y_1 - y_0) / h = 0")
    print("      Что дает: -y_0 + y_1 = 0")
    print("      Эта строка используется для первой строки в системе.")
    print("\n   b) Правое граничное условие (при x = B_INTERVAL = 1): y'(1) + y(1) = -0.75")
    print("      Аппроксимируется левой разностью для производной: (y_N - y_{N-1}) / h + y_N = -0.75")
    print("      Что дает: (-1/h) * y_{N-1} + ((h+1)/h) * y_N = -0.75")
    print("      Эта строка используется для последней строки в системе.")
    print("\nПолученная система линейных алгебраических уравнений решается методом прогонки (Tridiagonal Matrix Algorithm).")
    print("===================================================================\n")

    print("===================================================================")
    print("ПОРЯДКИ ТОЧНОСТИ МЕТОДОВ:")
    print("-> Метод конечных разностей:")
    print("   - Для используемой схемы конечных разностей: O(h^2) (глобальный порядок точности).")
    print("     Это означает, что при уменьшении шага h в 2 раза, ошибка должна уменьшаться в 4 раза.")
    print("===================================================================")

    xs_h = generate_grid_points(A_INTERVAL, B_INTERVAL, INITIAL_STEP_SIZE)
    num_points_h = len(xs_h)
    print(f"Сетка из {num_points_h} точек с шагом h = {INITIAL_STEP_SIZE}.")

    bc1_coeff_y0 = -1/INITIAL_STEP_SIZE
    bc1_coeff_y1 = 1/INITIAL_STEP_SIZE
    bc1_rhs = 0

    bc2_coeff_y_N_minus_1 = -1/INITIAL_STEP_SIZE
    bc2_coeff_y_N = (INITIAL_STEP_SIZE + 1)/INITIAL_STEP_SIZE
    bc2_rhs = -0.75

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
        print(f"x = {xs_h[i]:.5f}, Численное y(x) = {numerical_y_val:.5f}, Истинное y(x) = {true_y_val:.5f}, Абсолютная ошибка = {absolute_error:.16f}")
    print(f"\nМаксимальная абсолютная ошибка для шага h = {INITIAL_STEP_SIZE}: {total_error_h:.16f}")

    print("\n===================================================================")
    print("Расчет для шага в два раза короче (h/2) для оценки погрешности по Рунге:")
    half_step_size = INITIAL_STEP_SIZE / 2
    xs_h2 = generate_grid_points(A_INTERVAL, B_INTERVAL, half_step_size)
    num_points_h2 = len(xs_h2)
    print(f"Сетка из {num_points_h2} точек с шагом h/2 = {half_step_size}.")

    bc1_coeff_y0_h2 = -1/half_step_size
    bc1_coeff_y1_h2 = 1/half_step_size
    bc2_coeff_y_N_minus_1_h2 = -1/half_step_size
    bc2_coeff_y_N_h2 = (half_step_size + 1)/half_step_size

    solution_y_h2 = solve_finite_difference(num_points_h2, xs_h2, half_step_size,
                                            A_b1=bc1_coeff_y0_h2, A_c1=bc1_coeff_y1_h2, b1=bc1_rhs,
                                            A_an=bc2_coeff_y_N_minus_1_h2, A_bn=bc2_coeff_y_N_h2, bn=bc2_rhs)

    max_error_h2 = 0.0
    for i in range(len(xs_h2)):
        true_y_val_h2 = get_true_solution(xs_h2[i])
        numerical_y_val_h2 = solution_y_h2[i]
        absolute_error_h2 = abs(numerical_y_val_h2 - true_y_val_h2)
        max_error_h2 = max(max_error_h2, absolute_error_h2)
    print(f"\nМаксимальная абсолютная ошибка для шага h/2 = {half_step_size}: {max_error_h2:.16f}")


    runge_error_value = calculate_runge_error(solution_y_h2, solution_y_h, 1)
    print(f"Апостериорная оценка погрешности по Рунге: {runge_error_value:.16f}")

    perform_condition_and_stiffness_analysis(A_INTERVAL, B_INTERVAL)

    print("\n===================================================================")
    print("ПРОВЕРКА ПО ТЕОРЕТИЧЕСКИМ ОЦЕНКАМ ПОГРЕШНОСТИ:")
    print("Примечание: Представленные теоретические оценки погрешности")
    print("  относятся к методу Рунге-Кутты 4-го порядка. Для метода конечных")
    print("  разностей, используемого здесь, порядок точности равен O(h^2).")
    print("  Формула, приводимая ниже, используется для сверки с требованиями эталонного теста.")
    print("===================================================================")

    try:
        M4_val = get_true_solution_4th_derivative(A_INTERVAL) 
        print(f"M4 (максимум модуля 4-й производной y(x) на [{A_INTERVAL}, {B_INTERVAL}]): {M4_val:.8e}")

        H_STEP = INITIAL_STEP_SIZE
        H_STEP_HALF = half_step_size

        max_error_rungekutta_h = total_error_h 

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
        print(f"\tФактическая ошибка:   E_actual = {max_error_h2:.8e}")
        if max_error_h2 <= theoretical_error_rungekutta_h2:
            print("\t-> Фактическая ошибка меньше или равна теоретической оценке (GOOD).")
        else:
            print("\t-> Фактическая ошибка превышает теоретическую оценку (CAUTION).")

    except ZeroDivisionError as e:
        print(f"Ошибка при вычислении M4 или теоретической оценки: {e}")
        print("    Невозможно выполнить теоретическую проверку погрешности из-за сингулярности 4-й производной.")
    except Exception as e:
        print(f"Произошла ошибка при расчете теоретической оценки погрешности: {e}")


    true_y_values_for_plot = np.array([get_true_solution(x) for x in xs_h])
    
    output_dir = '/home/snowwy/Desktop/MAI/_math/8_Численные_методы/numerical_methods/lab_4/src'
    plot_solution(xs_h, solution_y_h, true_y_values_for_plot, output_dir)

    print("\nЗавершение выполнения программы.")


if __name__ == "__main__":
    main()

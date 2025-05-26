# Метод простых итераций
import numpy as np


def print_matrix(matrix):
    for row in matrix:
        print(' '.join(f'{elem:.6f}' for elem in row))
    print()

def print_vector(vector):
    for val in vector:
        print(f"{val:.6f}")

    print()

def check_matrix_for_simple_iteration(matrix):
    n = len(matrix)
    diagonal_dominance_satisfied = True
    
    print("\n--- Проверка условий сходимости метода простой итерации ---")

    for i in range(n):
        if matrix[i][i] == 0:
            print(f"ОШИБКА: Диагональный элемент A[{i}][{i}] равен нулю. Метод простой итерации неприменим без перестановки строк.")
            return False

        row_sum = sum(abs(matrix[i][j]) for j in range(n) if j != i)

        if abs(matrix[i][i]) < row_sum:
            print(f"ПРЕДУПРЕЖДЕНИЕ: Для строки {i} отсутствует диагональное преобладание: |A[{i}][{i}]| = {abs(matrix[i][i]):.6f} < Сумма остальных элементов в строке = {row_sum:.6f}")
            diagonal_dominance_satisfied = False
            found = False
            for h in range(n): 
                if h == i:
                    continue
                temp_matrix = matrix.copy()
                temp_matrix[[i, h]] = temp_matrix[[h, i]] 
                
                diag_dom_after_swap = True
                for k in range(n):
                    current_row_sum_k = sum(abs(temp_matrix[k][t]) for t in range(n) if t != k)
                    if abs(temp_matrix[k][k]) < current_row_sum_k:
                        diag_dom_after_swap = False
                        break
                
                if diag_dom_after_swap:
                    matrix[:] = temp_matrix[:] 
                    print(f"ИНФОРМАЦИЯ: Строки {i} и {h} были переставлены для достижения диагонального преобладания для всей матрицы.")
                    diagonal_dominance_satisfied = True 
                    found = True
                    break
            
            if not found and not diagonal_dominance_satisfied: 
                print("ОШИБКА: Невозможно добиться диагонального преобладания путем перестановки строк.")
                return False
        else:
            print(f"ИНФОРМАЦИЯ: Для строки {i} условие диагонального преобладания выполняется: |A[{i}][{i}]| = {abs(matrix[i][i]):.6f} >= Сумма остальных элементов в строке = {row_sum:.6f}")

    if diagonal_dominance_satisfied:
        print("ИНФОРМАЦИЯ: Условие диагонального преобладания выполняется .")
    
    return diagonal_dominance_satisfied


def get_matrix_c_f_for_simple_iteration(matrix):
    n = len(matrix)
    result = np.zeros((n, n + 1), dtype=float)

    for i in range(n):
        for j in range(n):
            if i != j:
                result[i][j] = -matrix[i][j] / matrix[i][i]

        result[i][n] = matrix[i][n] / matrix[i][i]

    return result


def get_next_vector_of_x(matrix, vec):
    n = len(matrix)
    result_vec = np.zeros(n, dtype=float)

    for i in range(n):
        for j in range(n):
            result_vec[i] += vec[j] * matrix[i][j]

        result_vec[i] += matrix[i][n]

    return result_vec


def get_max_diff_for_two_vectors(first, second):
    if len(first) != len(second):
        raise ValueError("Векторы имеют разную длину!")

    return np.max(np.abs(np.subtract(first, second)))


def find_matrix_norm(matrix):
    n = len(matrix)
    max_norm = 0

    for i in range(n):
        row_sum = sum(abs(matrix[i][j]) for j in range(n))
        max_norm = max(max_norm, row_sum)

    return max_norm


def check_answer(matrix, n, results):
    print("\nМой ответ          Реальный ответ")

    for i in range(n):
        computed = sum(matrix[i][j] * results[j] for j in range(n))

        print(f"{computed:.6f}          {matrix[i][n]:.6f}")


def main():
    n = int(input('Введите размерность матрицы системы (одно положительное число): '))

    print(f"Введите матрицу A размером {n}x{n} построчно (элементы разделены пробелами):")
    A = np.array([list(map(np.float64, input().split())) for _ in range(n)], dtype=np.float64)

    print("Введите столбец свободных членов b (элементы через пробел):")
    b = np.array(list(map(np.float64, input().split())), dtype=np.float64)

    eps = float(input("Введите точность поиска решения (eps): "))

    print("\n--- Начальные проверки системы ---")
    if abs(np.linalg.det(A)) < 1e-10:
        print("ОШИБКА: Матрица вырождена. Система имеет бесконечно много решений, либо не имеет ни одного!")
        exit(0)
    else:
        print("ИНФОРМАЦИЯ: det(A) != 0, система имеет единственное решение.")

    A_extended = np.column_stack((A, b))

    if not check_matrix_for_simple_iteration(A_extended):
        print("ОШИБКА: Матрица не удовлетворяет условию диагонального преобладания или его нельзя добиться перестановкой строк. Метод простой итерации не может быть применен.")
        exit(1)

    matrix_c_f = get_matrix_c_f_for_simple_iteration(A_extended)
    print("\nМатрица Альфа и Вектор Бета:")
    print_matrix(matrix_c_f)

    vector_result = np.zeros(n)

    counter = 0
    matrix_norm = find_matrix_norm(matrix_c_f[:, :-1]) 

    print("\n--- Проверка условия сходимости по норме матрицы  ---")
    if matrix_norm < 1:
        print(f"ИНФОРМАЦИЯ: ||α|| = {matrix_norm:.6f} < 1. Достаточное условие сходимости по теореме 1 выполняется. Метод простой итерации гарантированно сходится.")
        
    else:
        print(f"ПРЕДУПРЕЖДЕНИЕ: ||α|| = {matrix_norm:.6f} >= 1. Достаточное условие сходимости НЕ выполняется. Метод простой итерации может не сходиться.")
        eigenvalues = np.linalg.eigvals(matrix_c_f[:, :-1])
        max_abs_eigenvalue = np.max(np.abs(eigenvalues))
        print(f"ИНФОРМАЦИЯ: |λ_max| = {max_abs_eigenvalue:.6f}")
        if max_abs_eigenvalue < 1:
            print("ИНФОРМАЦИЯ: |λ_max| < 1. Необходимое и достаточное условие сходимости выполняется. Метод простой итерации сходится.")
        else:
            print("ОШИБКА: |λ_max| >= 1. Необходимое и достаточное условие сходимости НЕ выполняется. Метод простой итерации расходится.")
            exit(1)

    e_k = 1 # Оценка погрешности
    x_prev = vector_result.copy() # x^(k-1)
    
    print("\n--- Выполнение итерационного процесса ---")
    while True:
        x_curr = get_next_vector_of_x(matrix_c_f, x_prev) # x^(k)

        diff_norm = get_max_diff_for_two_vectors(x_curr, x_prev) # ||x^(k) - x^(k-1)||

        if matrix_norm < 1: 
            e_k = (matrix_norm / (1 - matrix_norm)) * diff_norm
            if e_k <= eps:
                vector_result = x_curr
                break
        else: 
            if diff_norm <= eps: 
                vector_result = x_curr
                break

        x_prev = x_curr.copy()
        counter += 1
        
        if counter > 1000 and diff_norm > 1e-5: 
            print("ПРЕДУПРЕЖДЕНИЕ: Превышено максимальное количество итераций (1000). Метод, возможно, расходится или сходится очень медленно.")
            break


    print(f"\nРезультат:")
    print_vector(vector_result)

    check_answer(A_extended, n, vector_result)

    print(f"\nПодсчет количества итераций по теореме 2:")
    print("     Если ||α|| < 1, имеет место оценка погрешности:")
    print("||eps(k)|| <= ||α|| *||x(k) - x(k-1)|| / ( 1 - ||α|| )")
    print(f"\nКоличество итераций: {counter}.")


if __name__ == '__main__':
    main()
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
    
    print("\n--- Проверка условий применимости метода простой итерации ---")

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
                temp_row_i = temp_matrix[i].copy()
                temp_row_h = temp_matrix[h].copy()
                temp_matrix[i] = temp_row_h
                temp_matrix[h] = temp_row_i
                
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
                print("ИНФОРМАЦИЯ: Не удалось добиться диагонального преобладания путем перестановки строк. Метод может быть применен, но сходимость не гарантирована.")
                return False
        else:
            print(f"ИНФОРМАЦИЯ: Для строки {i} условие диагонального преобладания выполняется: |A[{i}][{i}]| = {abs(matrix[i][i]):.6f} >= Сумма остальных элементов в строке = {row_sum:.6f}")

    if diagonal_dominance_satisfied:
        print("ИНФОРМАЦИЯ: Условие диагонального преобладания выполняется для всей матрицы.")
    else:
        print("ПРЕДУПРЕЖДЕНИЕ: Условие диагонального преобладания не выполняется для всей матрицы. Это достаточное условие, его невыполнение не означает расходимости метода. Продолжаем работу.")
    
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
    print("\n--- Проверка решения ---")
    print("Мой ответ          Реальный ответ")

    for i in range(n):
        computed = sum(matrix[i][j] * results[j] for j in range(n))

        print(f"{computed:.6f}          {matrix[i][n]:.6f}")

def check_linear_independence(A):
    """
    Проверяет линейную зависимость/независимость строк матрицы A.
    """
    rank = np.linalg.matrix_rank(A)
    num_rows = A.shape[0]
    print(f"\n--- Проверка линейной независимости ---")
    print(f"Ранг матрицы A: {rank}")
    print(f"Количество строк в матрице A: {num_rows}")

    if rank == num_rows:
        print("ИНФОРМАЦИЯ: Система линейно независима (строки матрицы A не являются линейными комбинациями друг друга).")
        return True
    else:
        print("ОШИБКА: Система линейно зависима (хотя бы одна строка матрицы A является линейной комбинацией других).")
        return False


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

    if not check_linear_independence(A):
        print("Метод простых итераций не может быть применен к линейно зависимой системе, которая не имеет единственного решения.")
        exit(1) 

    A_extended = np.column_stack((A, b))

    diag_dominance_satisfied = check_matrix_for_simple_iteration(A_extended)

    matrix_c_f = get_matrix_c_f_for_simple_iteration(A_extended)
    print("\nМатрица Альфа и Вектор Бета:")
    print_matrix(matrix_c_f)

    vector_result = np.zeros(n)
    ##########     print("\n--- Выполнение итераций метода PI---")

    counter = 0
    matrix_norm = find_matrix_norm(matrix_c_f[:, :-1]) 

    print("\n--- Проверка условий сходимости метода простой итерации ---")
    if matrix_norm < 1:
        print(f"ИНФОРМАЦИЯ: ||α|| = {matrix_norm:.6f} < 1. Достаточное условие сходимости по норме матрицы выполняется. Метод простой итерации гарантированно сходится.")
        
    else:
        print(f"ПРЕДУПРЕЖДЕНИЕ: ||α|| = {matrix_norm:.6f} >= 1. Достаточное условие сходимости по норме матрицы НЕ выполняется. Невыполнение достаточности не означает расходимости.")
        
        eigenvalues = np.linalg.eigvals(matrix_c_f[:, :-1])
        max_abs_eigenvalue = np.max(np.abs(eigenvalues))
        print(f"ИНФОРМАЦИЯ: Спектральный радиус ρ(α) = |λ_max| = {max_abs_eigenvalue:.6f}")
        
        if max_abs_eigenvalue < 1:
            print("ИНФОРМАЦИЯ: ρ(α) < 1. Необходимое и достаточное условие сходимости выполняется. Метод простой итерации сходится.")
        else:
            print("ОШИБКА: ρ(α) >= 1. Необходимое и достаточное условие сходимости НЕ выполняется. Метод простой итерации расходится.")
            exit(1)

    e_k = 1 
    x_prev = vector_result.copy() 
    
    print("\n--- Выполнение итерационного процесса ---")
    while True:
        x_curr = get_next_vector_of_x(matrix_c_f, x_prev) 

        diff_norm = get_max_diff_for_two_vectors(x_curr, x_prev) 

        if matrix_norm < 1: 
            e_k = (matrix_norm / (1 - matrix_norm)) * diff_norm
            if e_k <= eps:
                vector_result = x_curr
                print(f"ИНФОРМАЦИЯ: Критерий остановки по Теореме 2 (оценка погрешности {e_k:.6f} <= eps {eps:.6f}) выполнен на итерации {counter}.")
                break
        else: 
            if diff_norm <= eps: 
                vector_result = x_curr
                print(f"ИНФОРМАЦИЯ: Критерий остановки ||x^(k) - x^(k-1)|| < epsilon ({diff_norm:.6f} < {eps:.6f}) выполнен на итерации {counter}.")
                print("ПРЕДУПРЕЖДЕНИЕ: Достаточное условие сходимости ||α|| < 1 не выполняется, достижение заданной точности не гарантируется.")
                break

        x_prev = x_curr.copy()
        counter += 1
        
        if counter > 10000 and diff_norm > 1e-5: 
            print("ПРЕДУПРЕЖДЕНИЕ: Превышено максимальное количество итераций (10000). Метод, возможно, сходится очень медленно или не сходится.")
            vector_result = x_curr 
            break

    ##########
    print(f"\n--- Результат ---")
    print_vector(vector_result)

    check_answer(A_extended, n, vector_result)

    print(f"\n--- Дополнительная информация ---")
    print(f"Количество итераций: {counter}.")
    if matrix_norm < 1:
        print("     Оценка погрешности по теореме 2: ||eps(k)|| <= ||α|| * ||x(k) - x(k-1)|| / ( 1 - ||α|| )")

if __name__ == '__main__':
    main()
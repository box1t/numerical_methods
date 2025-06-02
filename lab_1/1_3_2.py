# Метод Зейделя

import numpy as np

def print_matrix(matrix):
    for row in matrix:
        print(' '.join(f'{elem:.6f}' for elem in row))
    print()

def print_vector(vector):
    for val in vector:
        print(f"{val:.6f}")
    print()

def check_matrix_for_seidel_method(matrix_a_extended):
    n = len(matrix_a_extended)
    diagonal_dominance_satisfied = True
    
    print("\n--- Проверка условий применимости метода Зейделя ---")

    A_only = matrix_a_extended[:, :n] 

    for i in range(n):
        if A_only[i][i] == 0:
            print(f"ОШИБКА: Диагональный элемент A[{i}][{i}] равен нулю. Метод Зейделя неприменим без перестановки строк.")
            return False 

        row_sum = sum(abs(A_only[i][j]) for j in range(n) if j != i)

        if abs(A_only[i][i]) < row_sum:
            print(f"ПРЕДУПРЕЖДЕНИЕ: Для строки {i} отсутствует диагональное преобладание: |A[{i}][{i}]| = {abs(A_only[i][i]):.6f} < Сумма остальных элементов в строке = {row_sum:.6f}")
            diagonal_dominance_satisfied = False
            found_swap = False 
            
            for h in range(n): 
                if h == i:
                    continue
                temp_matrix_for_check = A_only.copy()
                temp_matrix_for_check[[i, h]] = temp_matrix_for_check[[h, i]]
                
                diag_dom_after_swap_for_all = True
                for k_check in range(n):
                    current_row_sum_k = sum(abs(temp_matrix_for_check[k_check][t]) for t in range(n) if t != k_check)
                    if abs(temp_matrix_for_check[k_check][k_check]) < current_row_sum_k:
                        diag_dom_after_swap_for_all = False
                        break
                
                if diag_dom_after_swap_for_all:
                    matrix_a_extended[[i, h]] = matrix_a_extended[[h, i]]
                    print(f"ИНФОРМАЦИЯ: Строки {i} и {h} были переставлены для достижения диагонального преобладания для всей матрицы.")
                    diagonal_dominance_satisfied = True 
                    found_swap = True
                    break 
            
            if not found_swap and not diagonal_dominance_satisfied: 
                print("ИНФОРМАЦИЯ: Не удалось добиться диагонального преобладания путем перестановки строк. Метод может быть применен, но сходимость не гарантирована.")
        else:
            print(f"ИНФОРМАЦИЯ: Для строки {i} условие диагонального преобладания выполняется: |A[{i}][{i}]| = {abs(A_only[i][i]):.6f} >= Сумма остальных элементов в строке = {row_sum:.6f}")

    if diagonal_dominance_satisfied:
        print("ИНФОРМАЦИЯ: Условие диагонального преобладания выполняется для всей матрицы.")
    else:
        print("ПРЕДУПРЕЖДЕНИЕ: Условие диагонального преобладания не выполняется для всей матрицы. Это достаточное условие, его невыполнение не означает расходимости метода. Продолжаем работу.")
    
    return True

def get_seidel_convergence_matrices(A_extended):
    n = A_extended.shape[0]
    A_matrix = A_extended[:, :n]
    b_vector = A_extended[:, n]

    B = np.tril(A_matrix, k=-1) 
    D = np.diag(np.diag(A_matrix)) 
    C = np.triu(A_matrix, k=1) 

    DL_inv = np.linalg.inv(D + B)

    C_seidel_for_convergence = -np.dot(DL_inv, C)
    f_seidel_for_convergence = np.dot(DL_inv, b_vector)

    return C_seidel_for_convergence, f_seidel_for_convergence

def get_next_vector_of_x_seidel(matrix_a_extended, x_prev):
    n = len(matrix_a_extended)
    x_curr = x_prev.copy() 

    A_matrix = matrix_a_extended[:, :n]
    b_vector = matrix_a_extended[:, n]

    for i in range(n):
        sum_val = 0
        for j in range(n):
            if j < i: 
                sum_val += A_matrix[i][j] * x_curr[j]
            elif j > i: 
                sum_val += A_matrix[i][j] * x_prev[j]
        
        x_curr[i] = (b_vector[i] - sum_val) / A_matrix[i][i]
    
    return x_curr


def get_max_diff(first, second):
    if len(first) != len(second):
        raise ValueError("Векторы имеют разную длину!")
    return np.max(np.abs(np.subtract(first, second)))

def find_matrix_norm_inf(matrix):
    n = len(matrix)
    max_norm = 0

    for i in range(n):
        row_sum = sum(abs(matrix[i][j]) for j in range(n))
        max_norm = max(max_norm, row_sum)

    return max_norm

def check_answer(original_matrix_extended, result):
    n = original_matrix_extended.shape[0]
    print("\n--- Проверка решения ---")
    print("Мой ответ          Реальный ответ")

    A_matrix = original_matrix_extended[:, :n]
    b_vector = original_matrix_extended[:, n]

    for i in range(n):
        lhs = sum(A_matrix[i][j] * result[j] for j in range(n))
        rhs = b_vector[i]

        print(f"{lhs:.6f}          {rhs:.6f}")

def check_linear_independence(A):
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
    A_input = np.array([list(map(np.float64, input().split())) for _ in range(n)], dtype=np.float64)

    print("Введите столбец свободных членов b (элементы через пробел):")
    b_input = np.array(list(map(np.float64, input().split())), dtype=np.float64)

    eps = float(input("Введите точность поиска решения (eps): "))

    print("\n--- Начальные проверки системы ---")
    if abs(np.linalg.det(A_input)) < 1e-10:
        print("ОШИБКА: Матрица вырождена. Система имеет бесконечно много решений, либо не имеет ни одного!")
        exit(0)
    else:
        print("ИНФОРМАЦИЯ: det(A) != 0, система имеет единственное решение.")

    if not check_linear_independence(A_input):
        print("Метод Зейделя не может быть применен к линейно зависимой системе, которая не имеет единственного решения.")
        exit(1) 

    A_extended_mutable = np.column_stack((A_input.copy(), b_input.copy()))

    if not check_matrix_for_seidel_method(A_extended_mutable):
        print("ОШИБКА: Невозможно применить метод Зейделя из-за нулевого диагонального элемента после всех попыток перестановки.")
        exit(1)

    C_seidel_for_convergence, f_seidel_for_convergence_unused = get_seidel_convergence_matrices(A_extended_mutable) 
    print("\nМатрица C (итераций Зейделя, для анализа сходимости):")
    print_matrix(C_seidel_for_convergence)

    vector_result = np.zeros(n, dtype=float) 
    counter = 0

    matrix_norm_seidel = find_matrix_norm_inf(C_seidel_for_convergence) 

    print("\n--- Проверка условий сходимости метода Зейделя ---")
    if matrix_norm_seidel < 1:
        print(f"ИНФОРМАЦИЯ: ||α|| = {matrix_norm_seidel:.6f} < 1. Достаточное условие сходимости по норме матрицы выполняется. Метод Зейделя гарантированно сходится.")
        
    else:
        print(f"ПРЕДУПРЕЖДЕНИЕ: ||α|| = {matrix_norm_seidel:.6f} >= 1. Достаточное условие сходимости по норме матрицы НЕ выполняется. Невыполнение достаточности не означает расходимости.")
        
        eigenvalues = np.linalg.eigvals(C_seidel_for_convergence)
        max_abs_eigenvalue = np.max(np.abs(eigenvalues))
        print(f"ИНФОРМАЦИЯ: Спектральный радиус ρ(α) = |λ_max| = {max_abs_eigenvalue:.6f}")
        
        if max_abs_eigenvalue < 1:
            print("ИНФОРМАЦИЯ: ρ(α) < 1. Необходимое и достаточное условие сходимости выполняется. Метод Зейделя сходится.")
        else:
            print("ОШИБКА: ρ(α) >= 1. Необходимое и достаточное условие сходимости НЕ выполняется. Метод Зейделя расходится.")
            exit(1)

    e_k = 1 
    x_prev = vector_result.copy() 
    
    print("\n--- Выполнение итерационного процесса ---")
    while True:
        x_curr = get_next_vector_of_x_seidel(A_extended_mutable, x_prev) 

        diff_norm = get_max_diff(x_curr, x_prev) 

        if matrix_norm_seidel < 1: 
            e_k = (matrix_norm_seidel / (1 - matrix_norm_seidel)) * diff_norm
            if e_k <= eps:
                vector_result = x_curr
                print(f"ИНФОРМАЦИЯ: Критерий остановки по Теореме 2 (оценка погрешности {e_k:.6f} <= eps {eps:.6f}) выполнен на итерации {counter}.")
                break
        else: 
            if diff_norm <= eps: 
                vector_result = x_curr
                print(f"ИНФОРМАЦИЯ: Критерий остановки ||x^(k) - x^(k-1)|| < epsilon ({diff_norm:.6f} < {eps:.6f}) выполнен на итерации {counter}.")
                print("ПРЕДУПРЕЖДЕНИЕ: Достаточное условие сходимости ||α|| < 1 не выполняется. Достижение заданной точности не гарантируется через оценку по Теореме 2, но итерации остановились по разности между приближениями.")
                break

        x_prev = x_curr.copy()
        counter += 1
        
        if counter > 10000 and diff_norm > 1e-5: 
            print("ПРЕДУПРЕЖДЕНИЕ: Превышено максимальное количество итераций (10000). Метод, возможно, сходится очень медленно или не сходится.")
            vector_result = x_curr 
            break
            
    print(f"\n--- Результат ---")
    print_vector(vector_result)

    check_answer(A_extended_mutable, vector_result) 
    print(f"\n--- Дополнительная информация ---")
    print(f"Количество итераций: {counter}.")
    if matrix_norm_seidel < 1:
        print("     Оценка погрешности по теореме 2: ||eps(k)|| <= ||C|| * ||x(k) - x(k-1)|| / ( 1 - ||α|| )")


if __name__ == '__main__':
    main()
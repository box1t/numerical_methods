import numpy as np

np.set_printoptions(precision=6, suppress=True)

def print_matrix(matrix_name, matrix):
    """
    Выводит матрицу с заданным именем.
    """
    print(f"{matrix_name}:")
    print(np.round(matrix, 6))
    print()

def identity_matrix(n):
    """
    Создает единичную матрицу размера n x n.
    """
    return np.eye(n)

def zeros(rows, cols):
    """
    Создает матрицу, заполненную нулями, размера rows x cols.
    """
    return np.zeros((rows, cols))

def LU_decompose_with_pivot(A):
    """
    Выполняет LUP-разложение матрицы A.
    Возвращает матрицы L, U и P (матрица перестановок).
    """
    n = len(A)
    U = np.copy(A).astype(float)
    L = identity_matrix(n).astype(float)
    P = identity_matrix(n).astype(float)
    num_swaps = 0

    for k in range(n):
        # Выбор главного элемента
        max_val = 0.0
        max_index = k
        for i in range(k, n):
            if abs(U[i][k]) > max_val:
                max_val = abs(U[i][k])
                max_index = i

        if max_val == 0:
            raise ValueError("Матрица вырождена, поскольку найден нулевой диагональный элемент после выбора главного элемента.")

        if max_index != k:
            # Меняем строки в U
            U[[k, max_index]] = U[[max_index, k]]
            # Меняем строки в L (для элементов до k-го столбца)
            L[[k, max_index], :k] = L[[max_index, k], :k]
            # Меняем строки в P
            P[[k, max_index]] = P[[max_index, k]]
            num_swaps += 1
            print(f"Меняем строки {k} и {max_index} (обмен главного элемента)")
            print_matrix("U после перестановки", U)
            print_matrix("L после перестановки", L)
            print_matrix("P после перестановки", P)


        for i in range(k + 1, n):
            if U[k][k] == 0:
                raise ValueError("Деление на ноль при LU-разложении (нулевой опорный элемент).")
            factor = U[i][k] / U[k][k]
            L[i][k] = factor
            U[i, k:] -= factor * U[k, k:]
    return L, U, P, num_swaps

def lu_solve(L, U, P, b):
    """
    Решает систему P A x = b (L U x = P b)
    Использует прямой ход для Ly = Pb и обратный ход для Ux = y.
    """
    n = len(L)
    Pb = P @ b

    # Прямой ход: Ly = Pb
    y = np.zeros(n)
    for i in range(n):
        sum_Lk_yk = 0
        for k in range(i):
            sum_Lk_yk += L[i][k] * y[k]
        y[i] = (Pb[i] - sum_Lk_yk) / L[i][i] # L[i][i] всегда 1 для L из LUP

    # Обратный ход: Ux = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if U[i][i] == 0:
            raise ValueError("Невозможно выполнить обратный ход, деление на ноль.")
        sum_Uk_xk = 0
        for k in range(i + 1, n):
            sum_Uk_xk += U[i][k] * x[k]
        x[i] = (y[i] - sum_Uk_xk) / U[i][i]
    return x

def compute_determinant(U, num_swaps):
    """
    Вычисляет определитель матрицы A, используя U-матрицу и количество перестановок.
    det(A) = det(P)^-1 * det(L) * det(U) = det(P) * det(U)
    det(P) = (-1)^num_swaps
    det(L) = 1 (так как L имеет 1 на диагонали)
    """
    n = len(U)
    det_U = 1.0
    for i in range(n):
        det_U *= U[i][i]
    det_P = (-1) ** num_swaps
    det_A = det_P * det_U
    return det_A

def compute_inverse(L, U, P):
    """
    Вычисляет обратную матрицу A^-1, используя L, U и P.
    A^-1 = (LU)^-1 P = U^-1 L^-1 P
    """
    n = len(L)
    inverse = zeros(n, n)
    I = identity_matrix(n)
    for i in range(n):
        e = I[:, i] # i-й столбец единичной матрицы
        x = lu_solve(L, U, P, e) # Решаем Ax = e_i
        for j in range(n):
            inverse[j][i] = x[j]
        print(f"Решение для столбца {i} обратной матрицы: {x}")
    return inverse

def check_solution(A, b, x):
    """
    Проверяет решение системы Ax = b.
    """
    b_calc = A @ x
    print("\nМой ответ         Исходный вектор b")
    for bi, br in zip(b_calc, b):
        print(f"{bi:.6f}    {br:.6f}")

def check_linear_independence(A):
    """
    Проверяет линейную зависимость/независимость строк матрицы A.
    """
    rank = np.linalg.matrix_rank(A)
    num_rows = A.shape[0]
    print(f"\nРанг матрицы A: {rank}")
    print(f"Количество строк в матрице A: {num_rows}")

    if rank == num_rows:
        print("Система линейно независима (строки не являются линейными комбинациями друг друга).")
    else:
        print("Система линейно зависима (хотя бы одна строка является линейной комбинацией других).")


def main():
    n = int(input('Введите размерность матрицы системы (одно положительное число): '))

    print(f"Введите матрицу A размером {n}x{n} построчно (элементы разделены пробелами):")
    A = np.array([list(map(float, input().split())) for _ in range(n)], dtype=float)

    print("Введите столбец свободных членов b (элементы через пробел):")
    b = np.array(list(map(float, input().split())), dtype=float)

    print_matrix("Исходная матрица A", A)
    print("Исходный вектор b:", b)

    try:
        L, U, P, num_swaps = LU_decompose_with_pivot(A)
        print("\n--- Результаты LUP-разложения ---")
        print_matrix("Матрица L", L)
        print_matrix("Матрица U", U)
        print_matrix("Финальная перестановочная матрица P", P)


        print("\nПроверка L * U:")
        print_matrix("L * U", L @ U)
        print("\nПроверка P @ A:")
        print_matrix("P @ A", P @ A)
        print("\nПроверка L @ U == P @ A:")
        print_matrix("L @ U - P @ A", L @ U - P @ A)


        print("\n--- Решение системы Ax=b с помощью LU-разложения ---")
        x = lu_solve(L, U, P, b)
        print("Вектор решения x:")
        for x_i in x:
            print(f"{x_i:.6f}")

        det_A = compute_determinant(U, num_swaps)
        print(f"\ndet(A): {det_A:.6f}")

        # Проверка на линейную зависимость/независимость
        check_linear_independence(A)

        if np.abs(det_A) < 1e-9: # Используем небольшой допуск для сравнения с нулем
            print("Матрица A вырождена, обратной не существует.")
            return

        print("\n--- Вычисление обратной матрицы A^-1 ---")
        A_inv = compute_inverse(L, U, P)
        print_matrix("Обратная матрица A^-1", A_inv)

        print("Проверка A * A^-1:")
        print_matrix("A * A^-1", A @ A_inv)

        print("Проверка A^-1 * A:")
        print_matrix("A^-1 * A", A_inv @ A)

        print("\n--- Проверка решения системы ---")
        check_solution(A, b, x)


    except ValueError as e:
        print(f"\nОшибка: {e}")
        print("Матрица вырождена или возникла проблема при разложении/решении.")
    except Exception as e:
        print(f"\nПроизошла непредвиденная ошибка: {e}")


if __name__ == "__main__":
    main()
import numpy as np

np.set_printoptions(precision=6, suppress=True)


def LU_decompose(A):
    n = len(A)
    L = [[0 for _ in range(n)] for _ in range(n)]
    U = np.copy(A)

    for k in range(0, n):
        for i in range(k, n):
            L[i][k] = U[i][k] / U[k][k]

        for i in range(k + 1, n):
            for j in range(n):
                U[i][j] -= L[i][k] * U[k][j]

    return L, U


def LU_method(A: np.ndarray, b: np.ndarray):
    A = A.copy()
    b = b.copy()
    n = A.shape[0]
    amount_replace = 0

    for i in range(n):
        max_row = i + np.argmax(np.abs(A[i:, i]))
        if abs(A[max_row, i] - A[i, i]) > 1e-8:
            A[[i, max_row]] = A[[max_row, i]]
            b[[i, max_row]] = b[[max_row, i]]
            amount_replace += 1

        for k in range(i + 1, n):
            coefficient = A[k, i] / A[i, i]
            A[k, i:] -= coefficient * A[i, i:]
            b[k] -= coefficient * b[i]

    # Обратный ход (решение Ax = b через LU)
    x = np.zeros(n, dtype=np.float64)

    # Прямой ход для Ly = b (L неявно в нижнем треугольнике A)
    y = np.zeros(n, dtype=np.float64)
    for i in range(n):
        y[i] = b[i] - np.dot(A[i, :i], y[:i])

    # Обратный ход для Ux = y
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]

    return x, amount_replace, A


def get_inverse_matrix(A):
    n = A.shape[0]
    identity = np.eye(n)
    inv = np.zeros_like(A, dtype=np.float64)

    for i in range(n):
        e = identity[:, i]
        inv[:, i], _, _ = LU_method(A, e)

    return inv


def check_solution(A, b, x):
    b_calc = A @ x
    print("My answer    Real b")
    for bi, br in zip(b_calc, b):
        print(f"{bi:.6f}    {br:.6f}")


def print_matrix(name, matrix):
    print(f"{name}:")
    print(np.round(matrix, 6))
    print()


def main():
    n = int(input('Введите размерность матрицы системы (одно положительное число): '))

    print(f"Введите матрицу A размером {n}x{n} построчно (элементы разделены пробелами):")
    A = np.array([list(map(np.float64, input().split())) for _ in range(n)], dtype=np.float64)

    print("Введите столбец свободных членов b (элементы через пробел):")
    b = np.array(list(map(np.float64, input().split())), dtype=np.float64)

    if abs(np.linalg.det(A)) < 1e-10:
        print("Матрица вырождена. Система имеет бесконечно много решений, либо не имеет ни одного. Обратной матрицы не существует")
        exit(0)

    L, U = LU_decompose(A)
    x, amount_replace, lu_matrix = LU_method(A, b)
    print("\nLU-разложение:")
    print_matrix("L matrix", L)
    print_matrix("U matrix", U)

    print("Проверка L * U:")
    print_matrix("L * U", L @ U)

    print("Вектор решения x:")
    for x_i in x:
        print(f"{x_i:.6f}")

    det = ((-1) ** amount_replace) * np.prod(np.diag(lu_matrix))
    print(f"\ndet(A): {det:.6f}")
    if np.abs(det) < 1e-10:
        print("Матрица A вырождена, обратной не существует.")
        return

    A_inv = get_inverse_matrix(A)
    print_matrix("Обратная матрица A^-1", A_inv)

    print("Проверка A * A^-1:")
    print_matrix("A * A^-1", A @ A_inv)

    print("Проверка A^-1 * A:")
    print_matrix("A^-1 * A", A_inv @ A)

    print("Проверка решения системы:")
    check_solution(A, b, x)


if __name__ == "__main__":
    main()



# A = np.array([
#     [8, 8, -5, -8],
#     [8, -5, 9, -8],
#     [5, -4, -6, -2],
#     [8, 3, 6, 6],
# ], np.float32)
# b = np.array([13, 38, 14, -95], np.float32)

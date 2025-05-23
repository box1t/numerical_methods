### Simple Iterations Method

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

    for i in range(n):
        if matrix[i][i] == 0:
            return False

        row_sum = sum(abs(matrix[i][j]) for j in range(n) if j != i)

        if abs(matrix[i][i]) < row_sum:
            found = False

            for h in range(i + 1, n):
                row_sum_h = sum(abs(matrix[h][t]) for t in range(n) if t != h)

                if abs(matrix[h][i]) > row_sum and abs(matrix[h][h]) > row_sum_h:
                    matrix[i], matrix[h] = matrix[h], matrix[i]
                    found = True
                    break

            if not found:
                return False

    return True


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
        raise ValueError("Vectors have different lengths!")

    return np.max(np.abs(np.subtract(first, second)))


def find_matrix_norm(matrix):
    n = len(matrix)
    max_norm = 0

    for i in range(n):
        row_sum = sum(abs(matrix[i][j]) for j in range(n))
        max_norm = max(max_norm, row_sum)

    return max_norm


def check_answer(matrix, n, results):
    print("\nMy answer    Real answer")

    for i in range(n):
        computed = sum(matrix[i][j] * results[j] for j in range(n))

        print(f"{computed:.6f} {matrix[i][n]:.6f}")


def main():
    n = int(input('Введите размерность матрицы системы (одно положительное число): '))

    print(f"Введите матрицу A размером {n}x{n} построчно (элементы разделены пробелами):")
    A = np.array([list(map(np.float64, input().split())) for _ in range(n)], dtype=np.float64)

    print("Введите столбец свободных членов b (элементы через пробел):")
    b = np.array(list(map(np.float64, input().split())), dtype=np.float64)

    eps = float(input("Введите точность поиска решения (eps): "))

    if abs(np.linalg.det(A)) < 1e-10:
        print("ERROR: Матрица вырождена. Система имеет бесконечно много решений, либо не имеет ни одного!")
        exit(0)

    A = np.column_stack((A, b))

    if not check_matrix_for_simple_iteration(A):
        print("ERROR: Матрица не имеет диагонального преобладания")
        exit(1)

    matrix_c_f = get_matrix_c_f_for_simple_iteration(A)
    print("\nMatrix Alpha and Beta:")
    print_matrix(matrix_c_f)

    vector_result = np.zeros(n)

    diff_prev = 0.0
    diff_norm = 0.0
    counter = 0
    matrix_norm = find_matrix_norm(matrix_c_f)
    e_k = 1

    while e_k > eps :
        tmp = vector_result.copy()
        vector_result = get_next_vector_of_x(matrix_c_f, vector_result)

        diff_norm = get_max_diff_for_two_vectors(vector_result, tmp)
        if matrix_norm >= 1:
            print(f"This system cannot be solved by the simple iteration method - matrix_norm = {matrix_norm}")
            e_k = diff_norm
        else:
            e_k = (matrix_norm / (1 - matrix_norm)) * diff_norm

        counter += 1

    print(f"The result is:")
    print_vector(vector_result)

    check_answer(A, n, vector_result)

    print(f"Amount of iterations: {counter}")


if __name__ == '__main__':
    main()

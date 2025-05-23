import numpy as np


def float64_eq(a, b) -> bool:
    return abs(a - b) < 1e-8


def check_input_matrix(matrix: np.ndarray) -> None:
    matrix_size = matrix.shape[0]

    for i in range(matrix_size):
        for j in range(matrix_size):
            if abs(i - j) > 1 and not float64_eq(matrix[i, j], 0):
                print("ERROR: Метод Прогонки не применим - матрица не является трехдиагональной!")
                exit(1)


def check_diagonals(a, b, c) -> None:
    is_failed = False

    if float64_eq(c[0], 0) or float64_eq(b[0], 0) or float64_eq(b[-1], 0) or float64_eq(a[-1], 0):
        is_failed = True
        print("First or last line has incorrect values")

    is_b_bigger = 0
    for i in range(1, len(b) - 1):
        if is_failed:
            break

        if abs(b[i]) > abs(a[i]) + abs(c[i]):
            is_b_bigger += 1

        if abs(b[i]) < abs(a[i]) + abs(c[i]):
            is_failed = True
            print("One or more rows failed check: abs(b) < abs(a) + abs(c)")

        if float64_eq(a[i], 0):
            is_failed = True
            print("One or more values in A vector equals zero")

        if float64_eq(c[i], 0):
            is_failed = True
            print("One or more values in C vector equals zero")

    if is_b_bigger == 0:
        is_failed = True
        print("One or more rows failed check: abs(b) < abs(a) + abs(c)")

    if is_failed:
        print("ERROR: Метод Прогонки не применим - матрица некорректна!")
        exit(1)


def extract_diagonals(matrix: np.ndarray) -> tuple[list[np.float64], list[np.float64], list[np.float64]]:
    n = len(matrix)

    a: list[np.float64] = [np.float64(0.0)] * n
    b: list[np.float64] = [np.float64(0.0)] * n
    c: list[np.float64] = [np.float64(0.0)] * n

    for i in range(n):
        b[i] = matrix[i][i]

        if i > 0:
            a[i] = matrix[i][i - 1]
        if i < n - 1:
            c[i] = matrix[i][i + 1]

    return a, b, c


def check_answer(
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        d: np.ndarray,
        results: np.ndarray
) -> None:
    n = len(results) - 1
    answer = [0.0] * len(results)

    print("\nMy answer         Real answer")

    answer[0] = results[0] * b[0] + results[1] * c[0]
    print(f"{answer[0]:.6f}         {d[0]:.6f}")

    for i in range(1, n):
        answer[i] = results[i - 1] * a[i] + b[i] * results[i] + c[i] * results[i + 1]
        print(f"{answer[i]:.6f}         {d[i]:.6f}")

    answer[n] = results[n - 1] * a[n] + b[n] * results[n]
    print(f"{answer[n]:.6f}         {d[n]:.6f}")


def main():
    n = int(input('Введите размерность матрицы системы (одно положительное число): '))

    print(f"Введите матрицу A размером {n}x{n} построчно (элементы разделены пробелами):")
    A = np.array([list(map(np.float64, input().split())) for _ in range(n)], dtype=np.float64)

    print("Введите столбец свободных членов b (элементы через пробел):")
    b = np.array(list(map(np.float64, input().split())), dtype=np.float64)

    check_input_matrix(A)

    a_diag, b_diag, c_diag = extract_diagonals(A)

    check_diagonals(a_diag, b_diag, c_diag)

    x: list[np.float64] = [np.float64(0.0)] * n
    p: list[np.float64] = [np.float64(0.0)] * n
    q: list[np.float64] = [np.float64(0.0)] * n

    p[0] = -(c_diag[0] / b_diag[0])
    q[0] = b[0] / b_diag[0]

    for i in range(1, n):
        divider = b_diag[i] + a_diag[i] * p[i - 1]

        if abs(divider) < 1e-8:
            print("ERROR: Эта система не может быть решена Методом Прогонки - происходит деление на 0!")
            exit(1)

        p[i] = -c_diag[i] / divider
        q[i] = (b[i] - a_diag[i] * q[i - 1]) / divider

    for val in p:
        if abs(val) > 1:
            print("ERROR: One or more elements of vector P evaluated out of range (-1; 1)!")
            exit(1)

    x[-1] = q[-1]

    for i in range(n - 2, -1, -1):
        x[i] = p[i] * x[i + 1] + q[i]

    print("\nAnswer:")
    for x_i in x:
        print(f"{x_i:.6f}")

    check_answer(a_diag, b_diag, c_diag, b, x)


if __name__ == '__main__':
    main()

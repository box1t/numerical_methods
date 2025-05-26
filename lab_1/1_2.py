import numpy as np


def float64_eq(a, b) -> bool:
    return abs(a - b) < 1e-8

def check_input_matrix(matrix: np.ndarray) -> None:
    matrix_size = matrix.shape[0]

    for i in range(matrix_size):
        for j in range(matrix_size):
            if abs(i - j) > 1 and not float64_eq(matrix[i, j], 0):
                print("ОШИБКА: Метод Прогонки неприменим - матрица не является трехдиагональной!")
                exit(1)

def check_diagonals(a, b, c) -> None:
    is_failed = False
    n = len(b)

    print("Проверка на достаточное условие корректности (диагональное преобладание) (|b_i| >= |a_i| + |c_i|):")
    for i in range(n):
        val_a = a[i] if i > 0 else 0.0
        val_c = c[i] if i < n - 1 else 0.0

        if abs(b[i]) < abs(val_a) + abs(val_c) and not float64_eq(abs(b[i]), abs(val_a) + abs(val_c)):
            is_failed = True
            print(f"  Строка {i+1}: |b[{i}]| ({abs(b[i]):.6f}) < |a[{i}]| ({abs(val_a):.6f}) + |c[{i}]| ({abs(val_c):.6f}). Условие не выполнено!")
            break
        else:
            print(f"  Строка {i+1}: |b[{i}] ({abs(b[i]):.6f}) >= |a[{i}]| ({abs(val_a):.6f}) + |c[{i}]| ({abs(val_c):.6f}). Условие выполнено.")

    if is_failed:
        print("ОШИБКА: Метод Прогонки может быть некорректен - не выполнено достаточное условие корректности!")

    # требования к b_i != 0 из условия b_i - a_i P_i != 0
    for i in range(n):
        if float64_eq(b[i], 0):
            is_failed = True
            print(f"ОШИБКА: Элемент B[{i}] равен нулю. Метод Прогонки неприменим!")
            break

    if is_failed:
        print("ОШИБКА: Метод Прогонки неприменим - матрица некорректна!")
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
        a: list[np.float64],  
        b: list[np.float64],  
        c: list[np.float64],  
        d: np.ndarray,
        results: np.ndarray
) -> None:
    n = len(results) - 1
    answer = [0.0] * len(results)

    print("\nМой ответ         Исходный вектор b")

    # Проверка для первой строки: b_0 * x_0 + c_0 * x_1 = d_0
    answer[0] = results[0] * b[0] + results[1] * c[0]
    print(f"{answer[0]:.6f}         {d[0]:.6f}")

    # Проверка для средних строк: a_i * x_{i-1} + b_i * x_i + c_i * x_{i+1} = d_i
    for i in range(1, n):
        answer[i] = a[i] * results[i - 1] + b[i] * results[i] + c[i] * results[i + 1]
        print(f"{answer[i]:.6f}         {d[i]:.6f}")

    # Проверка для последней строки: a_n * x_{n-1} + b_n * x_n = d_n
    answer[n] = a[n] * results[n - 1] + b[n] * results[n]
    print(f"{answer[n]:.6f}         {d[n]:.6f}")


def main():
    n = int(input('Введите размерность матрицы системы (одно положительное число): '))

    print(f"Введите матрицу A размером {n}x{n} построчно (элементы разделены пробелами):")
    A = np.array([list(map(np.float64, input().split())) for _ in range(n)], dtype=np.float64)

    print("Введите столбец свободных членов b (элементы через пробел):")
    b_vector = np.array(list(map(np.float64, input().split())), dtype=np.float64) 
    check_input_matrix(A)

    a_diag, b_diag, c_diag = extract_diagonals(A)

    check_diagonals(a_diag, b_diag, c_diag)

    x: list[np.float64] = [np.float64(0.0)] * n
    p: list[np.float64] = [np.float64(0.0)] * n
    q: list[np.float64] = [np.float64(0.0)] * n

    # Условие корректности метода прогонки b_i - a_i * P_i != 0
    # P_0 = -c_0 / b_0
    if float64_eq(b_diag[0], 0):
        print("ОШИБКА: Элемент b[0] равен нулю. Метод Прогонки неприменим (деление на ноль при расчете P[0])!")
        exit(1)
    p[0] = -(c_diag[0] / b_diag[0])
    q[0] = b_vector[0] / b_diag[0]

    # Проверка устойчивости метода прогонки: |P_i| <= 1
    if abs(p[0]) > 1 and not float64_eq(abs(p[0]), 1):
        print(f"ОШИБКА: Метод Прогонки неустойчив - |P[0]| ({abs(p[0]):.6f}) > 1!")
        exit(1)
    elif float64_eq(abs(p[0]), 1):
        print(f"ВНИМАНИЕ: Метод Прогонки устойчив, но |P[0]| ({abs(p[0]):.6f}) = 1 (граница устойчивости).")
    else:
        print(f"Метод Прогонки устойчив: |P[0]| ({abs(p[0]):.6f}) <= 1.")


    for i in range(1, n):
        # Проверка условия корректности: b_i - a_i * P_{i-1} != 0
        divider = b_diag[i] + a_diag[i] * p[i - 1]

        if abs(divider) < 1e-8:
            print(f"ОШИБКА: Эта система не может быть решена Методом Прогонки - происходит деление на 0 при расчете P[{i}]!")
            exit(1)

        p[i] = -c_diag[i] / divider
        q[i] = (b_vector[i] - a_diag[i] * q[i - 1]) / divider

        # Проверка устойчивости метода прогонки: |P_i| <= 1
        if abs(p[i]) > 1 and not float64_eq(abs(p[i]), 1):
            print(f"ОШИБКА: Метод Прогонки неустойчив - |P[{i}]| ({abs(p[i]):.6f}) > 1!")
            exit(1)
        elif float64_eq(abs(p[i]), 1):
            print(f"ВНИМАНИЕ: Метод Прогонки устойчив, но |P[{i}]| ({abs(p[i]):.6f}) = 1 (граница устойчивости).")
        else:
            print(f"Метод Прогонки устойчив: |P[{i}]| ({abs(p[i]):.6f}) <= 1.")

    print("\nПроверка устойчивости и корректности метода прогонки пройдена успешно.")


    x[-1] = q[-1]

    for i in range(n - 2, -1, -1):
        x[i] = p[i] * x[i + 1] + q[i]

    print("\nОтвет:")
    for x_i in x:
        print(f"{x_i:.6f}")

    check_answer(a_diag, b_diag, c_diag, b_vector, x)


if __name__ == '__main__':
    main()
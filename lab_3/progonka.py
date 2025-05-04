
def extract_tridiagonals(n, A):
    b = [0] * n
    c = [0] * (n - 1)
    a = [0] * (n - 1)

    b[0] = A[0][0]
    if n > 1:
        c[0] = A[0][1]

    for i in range(1, n - 1):
        a[i - 1] = A[i][i - 1]
        b[i] = A[i][i]
        c[i] = A[i][i + 1]


    if n > 1:
        a[n - 2] = A[n - 1][n - 2]
        b[n - 1] = A[n - 1][n - 1]

    return a, b, c


def read_matrix():
    n = int(input("Введите размер системы (n): "))
    A = []
    f = []

    print("Введите строки матрицы (каждая строка должна содержать n коэффициентов и правую часть через пробел):")
    for i in range(n):
        row = list(map(float, input().split()))
        if len(row) != n + 1:
            raise ValueError(f"Ошибка: должна быть введена строка с {n + 1} числами, а не {len(row)}")
        A.append(row[:n])
        f.append(row[-1])
    return n, A, f


def is_tridiagonal(n, A, epsilon=1e-10):
    for i in range(n):
        for j in range(n):
            if abs(A[i][j]) > epsilon and (j < i - 1 or j > i + 1):
                return False
    return True


def scale_row(row, f_i):
    max_val = max(abs(x) for x in row)
    if max_val < 1e-10:
        if abs(f_i) < 1e-10:
            raise ValueError("Система имеет бесконечно много решений")
        else:
            raise ValueError("Система несовместна")
    scaling_factor = 1.0 / max_val
    return [x * scaling_factor for x in row], f_i * scaling_factor


def check_diagonal_dominance(a, b, c):
    n = len(b)
    is_dominant = True
    messages = []

    # Проверка первой строки
    if abs(b[0]) < abs(c[0]):
        is_dominant = False
        messages.append(f"Строка 0: |{b[0]:.2e}| < |{c[0]:.2e}|")

    # Проверка средних строк
    for i in range(1, n - 1):
        if abs(b[i]) < abs(a[i - 1]) + abs(c[i]):
            is_dominant = False
            messages.append(f"Строка {i}: |{b[i]:.2e}| < |{a[i - 1]:.2e}| + |{c[i]:.2e}|")

    # Проверка последней строки
    if n > 1 and abs(b[-1]) < abs(a[-1]):
        is_dominant = False
        messages.append(f"Строка {n - 1}: |{b[-1]:.2e}| < |{a[-1]:.2e}|")

    return is_dominant, messages


def solve_prog(a, b, c, d, n):
    alpha = [0.0] * n
    beta = [0.0] * n
    epsilon = 1e-12

    # Прямой ход
    try:
        # Первая строка
        denom = b[0]
        if abs(denom) < epsilon:
            if abs(d[0]) < epsilon:
                raise ValueError("Система имеет бесконечно много решений")
            else:
                raise ValueError("Система несовместна")
        alpha[0] = -c[0] / denom
        beta[0] = d[0] / denom

        # Промежуточные строки
        for i in range(1, n - 1):
            denom = b[i] + a[i - 1] * alpha[i - 1]
            numerator = d[i] - a[i - 1] * beta[i - 1]

            if abs(denom) < epsilon:
                if abs(numerator) < epsilon:
                    raise ValueError(f"Бесконечно много решений на строке {i + 1}")
                else:
                    raise ValueError(f"Несовместность на строке {i + 1}")

            alpha[i] = -c[i] / denom
            beta[i] = numerator / denom

        # Последняя строка
        denom = b[-1] + a[-1] * alpha[-2]
        numerator = d[-1] - a[-1] * beta[-2]

        if abs(denom) < epsilon:
            if abs(numerator) < epsilon:
                raise ValueError("Бесконечно много решений в последней строке")
            else:
                raise ValueError("Несовместность в последней строке")

        beta[-1] = numerator / denom

    except ZeroDivisionError as e:
        raise ValueError("Деление на ноль при вычислениях") from e

    # Обратный ход
    x = [0.0] * n
    x[-1] = beta[-1]
    for i in range(n - 2, -1, -1):
        x[i] = alpha[i] * x[i + 1] + beta[i]

    return x


def print_matrix(n, A, f):
    print("\nУсловие задачи:")
    print("Матрица коэффициентов и вектор правой части:")
    for i in range(n):
        row_str = " ".join(f"{A[i][j]:12.4e}" for j in range(n))
        print(f"{row_str} | {f[i]:12.4e}")


def check_solution(n, A, f, x):
    max_error = 0.0
    print("\nПроверка решения:")
    for i in range(n):
        sum_ax = sum(A[i][j] * x[j] for j in range(n))
        error = abs(sum_ax - f[i])
        max_error = max(max_error, error)
        print(f"Строка {i + 1}: Ошибка = {error:.4e}")
    print(f"\nМаксимальная ошибка: {max_error:.4e}")


if __name__ == "__main__":
    try:
        # Чтение и проверка матрицы
        n, A, f = read_matrix()
        print_matrix(n, A, f)

        if not is_tridiagonal(n, A):
            raise ValueError("Матрица не является трехдиагональной")

        # Масштабирование
        for i in range(n):
            A[i], f[i] = scale_row(A[i], f[i])

        # Извлечение диагоналей
        a, b, c = extract_tridiagonals(n, A)

        # Проверка условий
        is_dominant, messages = check_diagonal_dominance(a, b, c)
        if not is_dominant:
            print("\nПредупреждение: Отсутствует диагональное преобладание!")
            for msg in messages:
                print(msg)

        # Решение системы
        x = solve_prog(a, b, c, f, n)

        # Вывод результатов
        print("\nРешение системы:")
        for i, xi in enumerate(x):
            print(f"x[{i}] = {xi:.8e}")

        check_solution(n, A, f, x)

    except ValueError as e:
        print(f"\nОшибка: {str(e)}")
    except Exception as e:
        print(f"\nНеожиданная ошибка: {str(e)}")

"""
Ввод

5
-14 6 0 0 0 82
2 7 0 0 0 -51
0 -7 -18 -9 0 -46
0 0 2 -13 2 111
0 0 0 -7 -7 35

ЛЗ
5
-14 6 0 0 0 82
2 7 0 0 9 -51
0 -7 -18 -9 0 -46
0 0 2 -13 2 111
0 0 0 -7 -7 35

НУ
5
-14 6 0 0 0 82
2 7 0 0 0 -51
0 -7 -18 -9 0 -46
0 0 0 -14 -14 111
0 0 0 -7 -7 35

ДП
5
-14 69999 0 0 0 82
2 7 9999 0 0 -51
0 -789898 -18 -9 0 -46
0 0 1000 -14 -14 111
0 0 0 -7 -7 35

"""
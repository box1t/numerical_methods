import numpy as np

def check_eigenvectors(
        matrix,
        eigenvalues,
        eigenvectors
) -> bool:
    for i in range(matrix.shape[0]):
        check_vector = (eigenvectors[i] @ matrix) / eigenvalues[i]

        if not np.allclose(check_vector, eigenvectors[i]):
            return False

    return True

def is_symmetric_matrix(matrix) -> bool:
    matrix_size = matrix.shape[0]

    for i in range(matrix_size):
        for j in range(i + 1, matrix_size):
            if matrix[i, j] != matrix[j, i]:
                return False

    return True

# Позиция максимального элемента
def find_max_upper_element(X):
    n = X.shape[0]
    i_max, j_max = 0, 1
    max_elem = abs(X[0][1])

    for i in range(n):
        for j in range(i + 1, n):
            if abs(X[i][j]) > max_elem:
                max_elem = abs(X[i][j])
                i_max = i
                j_max = j

    return i_max, j_max

# Норма матрицы
def matrix_norm(X):
    norm = 0

    for i in range(len(X[0])):
        for j in range(i + 1, len(X[0])):
            norm += X[i][j] * X[i][j]

    return np.sqrt(norm)

# Метод вращений
def rotation_method(A_, eps):
    n = A_.shape[0]
    A = np.copy(A_)
    eigen_vectors = np.eye(n)
    iterations = 0

    while True:
        current_norm = matrix_norm(A)
        print(f"INFO: Проверка условия окончания итерационного процесса: t(A^({iterations+1})) = {current_norm}")
        if current_norm < eps:
            print(f"INFO: Условие окончания итерационного процесса t(A^({iterations+1})) < eps ({eps}) выполняется. Итерационный процесс останавливается.")
            break
        else:
            print(f"INFO: Условие окончания итерационного процесса t(A^({iterations+1})) > eps ({eps}) не выполняется. Итерационный процесс продолжается.")

        iterations += 1
        i_max, j_max = find_max_upper_element(A)
        if A[i_max][i_max] - A[j_max][j_max] == 0:
            phi = np.pi / 4
        else:
            phi = 0.5 * np.arctan(2 * A[i_max][j_max] / (A[i_max][i_max] - A[j_max][j_max]))

        # Матрица вращения
        U = np.eye(n)
        U[i_max][j_max] = -np.sin(phi)
        U[j_max][i_max] = np.sin(phi)
        U[i_max][i_max] = np.cos(phi)
        U[j_max][j_max] = np.cos(phi)

        A = U.T @ A @ U
        eigen_vectors = eigen_vectors @ U

    eigen_values = [A[i][i] for i in range(n)]

    return eigen_values, eigen_vectors, iterations


def calculate_results(matrix, eps):
    try:
        A = np.copy(matrix)

        # Выполняем метод вращений
        values, vectors, iters = rotation_method(A, eps)

        # Формируем результат
        result_text = f"\nСобственные значения:\n{', '.join(map(str, values))}\n"
        result_text += "Собственные вектора:\n"
        for i in range(vectors.shape[0]):
            result_text += f"СВ {i + 1}: {vectors[:, i]}\n"
        result_text += f"Итераций: {iters}\n"

        result_text += "\nПроверка:\n"

        result_text += f"След матрицы: {sum([matrix[i][i] for i in range(matrix.shape[0])])}\n"
        result_text += f"Сумма собственных значений: {sum(values)}\n"

        result_text += "\nПроверка уравнения A * v = λ * v для каждого собственного вектора:\n\n"

        for i in range(n):
            vec = vectors[:, i]
            lam = values[i]
            lhs = matrix @ vec
            rhs = lam * vec

            result_text += f"Собственный вектор {i + 1}:\n"
            result_text += f"A * v = {lhs}\n"
            result_text += f"λ * v = {rhs}\n"
            if np.allclose(lhs, rhs, atol=1e-6):
                result_text += "Проверка пройдена\n\n"
            else:
                result_text += "Проверка не пройдена\n\n"

        result_text += "\nv_i - (v_i * A) / λ_i\n"

        print(result_text)

    except ValueError as e:
        print("Ошибка", f"\nНеверный ввод данных: {str(e)}")
        exit(1)


if __name__ == '__main__':
    n = int(input('Введите размерность матрицы (одно положительное число): '))

    print(f'Введите матрицу {n}x{n}: ')
    input_matrix = np.empty((n, n), dtype=float)
    for row in range(n):
        input_matrix[row] = np.array(list(map(
                    float,
                    input().split())))

    eps = float(input("Введите точность поиска решения (eps): "))

    if not is_symmetric_matrix(input_matrix):
        print('INFO: Проверка на симметричность матрицы: Матрица не симметрична.')
        print('ERROR: Метод Вращений Якоби не применим - матрица не симметрична!')
        exit(1)
    else:
        print('INFO: Проверка на симметричность матрицы: Матрица симметрична.')
        print('INFO: Метод Вращений Якоби применим!')

    calculate_results(input_matrix, eps)
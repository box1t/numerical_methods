import numpy as np

def get_LU_decomposition(A):
    n = len(A)
    LU = np.copy(A)
    swaps = []
    for k in range(n): # обнуляемый столбец

        if (LU[k][k] == 0): # Ищем ненулевой элемент
            ind = -1
            for i in range(k + 1, n):
                if LU[i][i] != 0:
                    ind = i
                    break
            if ind == -1:
                continue
            LU[[k, ind]] = LU[[ind, k]] # Меняем местами строки если, нашли строку с ненулевым элементом
            swaps.append((k, ind))

        for i in range(k + 1, n): # текущая строка
            mu = LU[i][k] / LU[k][k]
            for j in range(k, n): # текущая столбец
                if (j == k):
                    LU[i][j] = mu
                else:
                    LU[i][j] -= mu * LU[k][j]

    return (LU, swaps)

def solve_system(LU, swaps, b):
    n = len(LU)
    b = np.copy(b)

    # Меняем строки в столбце свободных членов в соответствие с заменами строк в исходной матрице
    for swap in swaps:
        b[[swap[0], swap[1]]] = b[[swap[1], swap[0]]]

    # Lz = b
    z = np.zeros(n)
    for i in range(n):
        sum_ = sum([LU[i][j] * z[j] for j in range(i)])
        z[i] = b[i] - sum_

    # Ux = z
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        sum_ = sum([LU[i][j] * x[j] for j in range(n - 1, i, -1)])
        x[i] = (z[i] - sum_) / LU[i][i]

    return x

def get_system_determinant(LU, swaps):
    n = len(LU)
    det = 1
    for i in range(n):
        det *= LU[i][i]

    # Каждая замена строк исходной матрицы - смена знака у определителя
    if (len(swaps) % 2 == 1):
        det *= -1

    return det

def get_inverse_system_matrix(LU, swaps):
    n = len(LU)
    A = []
    for i in range(n):
        A.append(solve_system(LU, swaps, np.array([(1 if j == i else 0) for j in range(n)])))

    return np.column_stack(A)

def check_LU_decomposition(LU):
    n = len(LU)

    L = np.copy(LU)
    for i in range(n):
        L[i][i] = 1
        for j in range(i + 1, n):
            L[i][j] = 0

    U = np.copy(LU)
    for i in range(1, n):
        for j in range(i):
            U[i][j] = 0

    print("Матрица L:\n", L)
    print("Матрица U:\n", U)

    A = np.dot(L, U)
    print("Матрица A:\n", np.round(A, 2))

A = np.array([
    [8, 8, -5, -8],
    [8, -5, 9, -8],
    [5, -4, -6, -2],
    [8, 3, 6, 6],
], np.float32)
b = np.array([13, 38, 14, -95], np.float32)

(LU, swaps) = get_LU_decomposition(A)
print("LU разложение:\n", np.round(LU, 2))
print("Замены строк: ", swaps)

print("\nПроверка LU разложения:")
check_LU_decomposition(LU)

system_solution = solve_system(LU, swaps, b)
print("\nРешение системы: ", np.round(system_solution, 2))
print("Проверка решения системы: ", np.round(np.linalg.solve(A, b), 2))

system_determinant = get_system_determinant(LU, swaps)
print("\nОпределитель матрицы системы: ", round(system_determinant, 2))
print("Проверка определителя матрицы системы: ", round(np.linalg.det(A), 2))

inverse_system_matrixA = get_inverse_system_matrix(LU, swaps)
print("\nОбратная матрица системы:\n", np.round(inverse_system_matrixA, 4))
print("Проверка обратности:\n", np.round(np.dot(A, inverse_system_matrixA), 2))

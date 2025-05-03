import numpy as np

def recoverMatrix(A):
    n = len(A)
    B = np.zeros((n, n))
    for i in range(n):
        if (i == 0):
            B[i][0] = A[i][1]
            B[i][1] = A[i][2]
        elif (i == n - 1):
            B[i][n - 2] = A[i][0]
            B[i][n - 1] = A[i][1]
        else:
            B[i][i - 1] = A[i][0]
            B[i][i] = A[i][1]
            B[i][i + 1] = A[i][2]
    return B

def tridiagonalMatrixAlgorithm(A, b):
    n = len(b)

    # Вычисляем прогоночные коэффициенты
    P = np.empty((n))
    P[0] = -A[0][2] / A[0][1]
    Q = np.empty((n))
    Q[0] = b[0] / A[0][1]
    for i in range(n):
        P[i] = (-A[i][2]) / (A[i][1] + A[i][0] * P[i - 1])
        Q[i] = (b[i] - A[i][0] * Q[i - 1]) / (A[i][1] + A[i][0] * P[i - 1])

    # Обратный ход
    x = np.empty((n))
    x[n - 1] = Q[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = P[i] * x[i + 1] + Q[i]

    return x

n = int(input())

A = np.zeros((n, 3))
for i in range(n):
    a = list(map(int, input().split(" ")))

    if i == 0:
        A[i][0] = 0
        A[i][1] = a[0]
        A[i][2] = a[1]
    elif i == n - 1:
        A[i][0] = a[0]
        A[i][1] = a[1]
        A[i][2] = 0
    else:
        A[i][0] = a[0]
        A[i][1] = a[1]
        A[i][2] = a[2]

b = list(map(int, input().split(" ")))

x = tridiagonalMatrixAlgorithm(A, b)
print("Решение системы: ", x)

recA = recoverMatrix(A)
print("Восстановленная матрица:\n", recA)
print("Проверка решения: ", np.linalg.solve(recA, b))


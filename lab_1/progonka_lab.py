import numpy as np
tolerance = 1e-9

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

def progonka(A, b):
    n = len(b)
    if A.shape[0] != n or A.shape[1] != 3:
        print("Ошибка в progonka: Неверные размеры матрицы A.")
        return None

    P = np.empty((n))
    Q = np.empty((n))
    x = np.empty((n))

    if A[0][1] == 0:
         print("Ошибка в progonka: Деление на ноль при расчете P[0]. Главный диагональный элемент равен нулю.")
         return None
    P[0] = -A[0][2] / A[0][1]
    Q[0] = b[0] / A[0][1]

    for i in range(1, n):
        denominator = A[i][1] + A[i][0] * P[i-1]
        if abs(denominator) < tolerance:
             print(f"Ошибка в progonka: Деление на ноль на шаге прямого хода {i}. Знаменатель близок к нулю.")
             return None
        if i < n - 1:
            P[i] = -A[i][2] / denominator
        Q[i] = (b[i] - A[i][0] * Q[i-1]) / denominator

    x[n - 1] = Q[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = P[i] * x[i + 1] + Q[i]

    return x

# Весь код, который должен выполняться только при прямом запуске 1_2.py,
# переместите сюда:
if __name__ == "__main__":
    n = int(input("Введите размерность матрицы (n): "))

    A = np.zeros((n, 3))
    print("Введите коэффициенты матрицы A (три числа в строке для каждой строки):")
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

    print("Введите вектор b (n чисел через пробел):")
    b = list(map(int, input().split(" ")))

    x = progonka(A, b)
    if x is not None:
        print("Решение системы: ", x)

        recA = recoverMatrix(A)
        print("Восстановленная матрица:\n", recA)
        # Переводим b в numpy массив для np.linalg.solve
        b_np = np.array(b)
        print("Проверка решения: ", np.linalg.solve(recA, b_np))
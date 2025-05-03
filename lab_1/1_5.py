import numpy as np

def decompositionQR(A):
    n = len(A)
    A = np.copy(A)
    Q = np.eye(n)
    for i in range(n):

        # Строим вектор v
        v = np.zeros((n, 1))
        v[i] = A[i][i] + np.sign(A[i][i]) * np.sqrt(sum( [A[j][i] ** 2 for j in range(i, n)] ))
        for j in range(i + 1, n):
            v[j] = A[j][i]

        # Строим матрицу Хаусхолдера
        vTrans = np.transpose(v)
        H = np.eye(n) - 2 / np.dot(vTrans, v) * np.dot(v, vTrans)

        # Строим Q
        Q = np.dot(Q, H)

        # Строим Ak
        A = np.dot(H, A)

    return Q, A

def alrorithmQR(A, eps):
    n = len(A)
    A = np.copy(A)
    iter = 0
    lambdas = np.empty((n, 2))
    while (True):
        iter += 1
        Q, R = decompositionQR(A)
        A = np.dot(R, Q)

        flg = True
        skip = False
        # print(f"iter #{iter}")
        for i in range(n):
            if skip:
                skip = False
                continue

            if i < n - 1:
                D = A[i][i] ** 2 + A[i + 1][i + 1] ** 2 - 2 * A[i][i] * A[i + 1][i + 1] + 4 * A[i][i + 1] * A[i + 1][i]
                if D < 0:
                    re = (A[i][i] + A[i + 1][i + 1]) / 2
                    im = np.sqrt(-D) / 2
                    
                    # Критерий остановки для пары комплексно-сопряженных
                    lambda_ = np.sqrt(re ** 2 + im ** 2)
                    lambdaPrev = np.sqrt(lambdas[i][0] ** 2 + lambdas[i][1] ** 2)
                    # print(f"coord #{i}: abs(lambda_ - lambdaPrev) = {abs(lambda_ - lambdaPrev)}")
                    if iter > 1 and abs(lambda_ - lambdaPrev) > eps:
                        flg = False

                    lambdas[i][0] = re
                    lambdas[i][1] = im
                    lambdas[i + 1][0] = re
                    lambdas[i + 1][1] = -im

                    skip = True
                    continue

            lambdas[i][0] = A[i][i]
            lambdas[i][1] = 0
            # Критерий остановки для действительного значения
            sum_ = np.sqrt(sum([A[j][i] ** 2 for j in range(i + 1, n)]))
            # print(f"coord #{i}: sum = {sum_}")
            if sum_ > eps:
                flg = False

        if flg:
            break

    return lambdas, iter


n = int(input())

A = np.empty((n, n))
for i in range(n):
    A[i] = np.array(list(map(float, input().split(" "))))

eps = float(input())


lambdas, iter = alrorithmQR(A, eps)

print("Количество итераций: ", iter)
print()

w, v = np.linalg.eig(A)
print("Вычисленные собственные значения: ", ", ".join([f"{lambda_[0]} + {lambda_[1]}i" for lambda_ in lambdas]))

print("Проверка собственных значений: ", w)
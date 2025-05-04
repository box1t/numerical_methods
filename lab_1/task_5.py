import math
import cmath

def vec_norm(v):
    return math.sqrt(sum(x * x for x in v))

def vec_dot(u, v):
    return sum(ui * vi for ui, vi in zip(u, v))

def vec_scalar_mult(v, s):
    return [s * x for x in v]

def vec_sub(u, v):
    return [ui - vi for ui, vi in zip(u, v)]

def outer_product(u, v):
    n = len(u)
    m = len(v)
    return [[u[i] * v[j] for j in range(m)] for i in range(n)]

def identity_matrix(n):
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]

def matrix_copy(A):
    return [row[:] for row in A]

def mat_transpose(A):
    n = len(A)
    m = len(A[0])
    return [[A[i][j] for i in range(n)] for j in range(m)]

def mat_multiply(A, B):
    n = len(A)
    m = len(B[0])
    p = len(B)
    C = [[0 for _ in range(m)] for _ in range(n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                C[i][j] += A[i][k] * B[k][j]
    return C

def mat_subtract(A, B):
    n = len(A)
    m = len(A[0])
    return [[A[i][j] - B[i][j] for j in range(m)] for i in range(n)]

def mat_add(A, B):
    n = len(A)
    m = len(A[0])
    return [[A[i][j] + B[i][j] for j in range(m)] for i in range(n)]

def mat_scalar_mult(A, s):
    n = len(A)
    m = len(A[0])
    return [[s * A[i][j] for j in range(m)] for i in range(n)]

def householder_vector(a):
    norm_a = vec_norm(a)
    sign = -1 if a[0] >= 0 else 1
    v = a[:]
    v[0] -= sign * norm_a
    norm_v = vec_norm(v)
    if norm_v == 0:
        return v
    return [x / norm_v for x in v]

def apply_householder(A, u, k):
    n = len(A)
    m = len(A[0])
    for i in range(k, n):
        dot_val = 0
        for j in range(k, m):
            dot_val += u[i - k] * A[i][j]
        for j in range(k, m):
            A[i][j] -= 2 * u[i - k] * dot_val

def qr_decomposition(A):
    n = len(A)
    m = len(A[0])
    R = matrix_copy(A)
    Q = identity_matrix(n)
    for k in range(min(n, m)):
        a = [R[i][k] for i in range(k, n)]
        u = householder_vector(a)
        H = identity_matrix(n)
        for i in range(k, n):
            for j in range(k, n):
                H[i][j] -= 2 * u[i - k] * u[j - k]
        R = mat_multiply(H, R)
        Q = mat_multiply(Q, H)
    return Q, R

def qr_algorithm_eigenvalues(A, tol=1e-3, max_iter=1000):
    Ak = matrix_copy(A)
    n = len(Ak)
    prev_diag = [10 ** 10] * n
    for iteration in range(max_iter):
        Q, R = qr_decomposition(Ak)
        Ak = mat_multiply(R, Q)
        current_diag = [Ak[i][i] for i in range(n)]
        delta = max(abs(current_diag[i] - prev_diag[i]) for i in range(n))
        # print(f"Итерация {iteration + 1}, изменение диагонали: {delta}")
        if delta < tol:
            break
        prev_diag = current_diag[:]
    eigenvalues = []
    i = 0
    while i < n:
        if i == n - 1 or abs(Ak[i+1][i]) < tol:
            eigenvalues.append(Ak[i][i])
            i += 1
        else:
            a = Ak[i][i]
            b = Ak[i][i+1]
            c = Ak[i+1][i]
            d = Ak[i+1][i+1]
            tr = a + d
            det = a * d - b * c
            disc = (tr/2)**2 - det
            sqrt_disc = cmath.sqrt(disc)
            eig1 = tr/2 + sqrt_disc
            eig2 = tr/2 - sqrt_disc
            eigenvalues.append(eig1)
            eigenvalues.append(eig2)
            i += 2
    return eigenvalues


# def qr_algorithm(matrix, accuracy):
#     """
#     QR-алгоритм нахождения собственных значений матрицы
#     :param matrix: матрица
#     :param accuracy: точность поиска
#     :return: список собственных значений
#     """
#     mat_a = copy_matrix(matrix)
#     size = len(mat_a)
 
#     # Собственные значения и отметки о найденных решениях
#     eigenvalues = [complex(0, 0)] * size
#     found = [False] * size
 
#     maximum_number_of_iterations = 1000
 
#     for i in range(maximum_number_of_iterations):
#         mat_q, mat_r = qr(mat_a)
#         mat_a = multiply(mat_r, mat_q)
 
#         skip_iteration = False
 
#         # Поиск собственных значений
#         for j in range(size - 1):
#             if skip_iteration:
#                 skip_iteration = False
#                 continue
 
#             # Действительные значения
#             if math.sqrt(sum([mat_a[k][j] ** 2 for k in range(j + 1, size)])) <= accuracy:
#                 # Нашли действительное собственное значение
#                 found[j] = True
#                 eigenvalues[j] = complex(mat_a[j][j])
#                 continue
 
#             if found[j + 1]:
#                 # Следующий столбец уже занят => действительное значение, продолжить итерации
#                 continue
 
#             # Проверка дискриминанта
#             d = math.pow(mat_a[j][j] - mat_a[j + 1][j + 1], 2)\
#                 + 4.0 * mat_a[j][j + 1] * mat_a[j + 1][j]
#             if d >= 0.0:
#                 # Действительное значение
#                 continue
 
#             complex_value = complex(
#                 mat_a[j][j] + mat_a[j + 1][j + 1] / 2.0,
#                 math.sqrt(abs(d)) / 2.0
#             )
 
#             if abs(complex_value - eigenvalues[j]) <= accuracy:
#                 # Нашли комлексные собственные значения
#                 found[j] = True
#                 found[j + 1] = True
 
#             eigenvalues[j] = complex_value
#             eigenvalues[j + 1] = complex_value.conjugate()
 
#             # Пропустить следующий столбец (он занят сопряжённым)
#             skip_iteration = True
 
#         # Не все size-1 собственные значения найдены
#         if not all(found[:-1]):
#             continue
 
#         # Если последнее собственное значение не заполнено
#         # (не является сопряжённым комплексным)
#         if not found[size - 1]:
#             eigenvalues[size - 1] = complex(mat_a[size - 1][size - 1])
#             found[size - 1] = True
 
#         # Закончить итерации
#         break
 
#     return eigenvalues



def input_matrix():
    n = int(input("Введите размер матрицы (n): "))
    print("Введите строки матрицы, разделяя элементы пробелами:")
    matrix = []
    for i in range(n):
        row = list(map(float, input().split()))
        if len(row) != n:
            raise ValueError("Количество элементов в строке не соответствует размеру матрицы")
        matrix.append(row)
    return matrix

def check_solution(A, eigenvalues, tol=1e-3):
    trace_A = sum(A[i][i] for i in range(len(A)))
    sum_eigs = sum(eigenvalues)
    diff = abs(trace_A - sum_eigs)
    print(f"Проверка решения: |trace(A) - sum(eigenvalues)| = {diff}")
    if diff < tol:
        print("Проверка пройдена.")
    else:
        print("Проверка не пройдена.")

def main():
    A = input_matrix()
    print("\nИсходная матрица A:")
    for row in A:
        print(row)
    tol = float(input("\nВведите точность вычислений (например, 0.001): "))
    eigenvalues = qr_algorithm_eigenvalues(A, tol=tol)
    print("\nПриближённые собственные значения матрицы:")
    for i, val in enumerate(eigenvalues):
        print(f"λ[{i + 1}] = {val}")
    check_solution(A, eigenvalues, tol)

if __name__ == "__main__":
    main()

"""
Тесты
Ввод:

Обычные корни
3
-1 8 5
8 -4 4
2 9 -2

Комплексно-сопряжённые
3
1 3 1
1 1 4
4 3 1
"""
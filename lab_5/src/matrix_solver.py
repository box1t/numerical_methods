import numpy as np
from copy import deepcopy

# Настройки печати для отладки, если они используются в main
np.set_printoptions(precision=6, suppress=True)

def identity_matrix(n):
    return np.eye(n)

def zeros(rows, cols):
    return np.zeros((rows, cols))

def LU_decompose_with_pivot(A):
    """
    Выполняет LUP-разложение матрицы A.
    Возвращает матрицы L, U и P (матрица перестановок).
    Используется для полной матрицы.
    """
    n = len(A)
    U = np.copy(A).astype(float)
    L = identity_matrix(n).astype(float)
    P = identity_matrix(n).astype(float)
    num_swaps = 0

    for k in range(n):
        # Выбор главного элемента
        max_val = 0.0
        max_index = k
        for i in range(k, n):
            if abs(U[i][k]) > max_val:
                max_val = abs(U[i][k])
                max_index = i

        if max_val == 0:
            raise ValueError("Матрица вырождена, поскольку найден нулевой диагональный элемент после выбора главного элемента.")

        if max_index != k:
            # Меняем строки в U, L (до k-го столбца) и P
            U[[k, max_index]] = U[[max_index, k]]
            L[[k, max_index], :k] = L[[max_index, k], :k]
            P[[k, max_index]] = P[[max_index, k]]
            num_swaps += 1

        for i in range(k + 1, n):
            if U[k][k] == 0:
                raise ValueError("Деление на ноль при LU-разложении (нулевой опорный элемент).")
            factor = U[i][k] / U[k][k]
            L[i][k] = factor
            U[i, k:] -= factor * U[k, k:]
            
    return L, U, P, num_swaps

def lu_solve(L, U, P, b):
    """
    Решает систему P A x = b (L U x = P b)
    Использует прямой ход для Ly = Pb и обратный ход для Ux = y.
    """
    n = len(L)
    # Преобразуем b в вектор numpy
    b = np.array(b, dtype=float)
    
    Pb = P @ b

    # Прямой ход: Ly = Pb
    y = np.zeros(n)
    for i in range(n):
        sum_Lk_yk = 0
        # Используем numpy-операции для ускорения
        sum_Lk_yk = L[i, :i] @ y[:i]
        
        # L[i][i] всегда 1 для L из LUP
        y[i] = (Pb[i] - sum_Lk_yk) / L[i][i] 

    # Обратный ход: Ux = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if U[i][i] == 0:
            raise ValueError("Невозможно выполнить обратный ход, деление на ноль.")
            
        # Используем numpy-операции для ускорения
        sum_Uk_xk = U[i, i+1:] @ x[i+1:]
        
        x[i] = (y[i] - sum_Uk_xk) / U[i][i]
    return x.tolist() # Возвращаем список, чтобы соответствовать оригинальному solve()



class MATRIX_SOLVER:
    def __init__(self, a: list = [], b:list = [], c:list = [], d:list = []):
        # Преобразуем входные данные в numpy массивы для удобства работы
        self._a = np.array(a, dtype=float)
        self._b = np.array(b, dtype=float)
        self._c = np.array(c, dtype=float)
        self._d = np.array(d, dtype=float)
        self._n = len(d)

    def _build_tridiagonal_matrix(self):
        """
        Создает полную трехдиагональную матрицу A из векторов a, b, c.
        """
        A = np.zeros((self._n, self._n), dtype=float)
        
        # Заполнение главной диагонали
        np.fill_diagonal(A, self._b)
        
        # Заполнение верхней диагонали (c)
        if self._n > 1:
            # c[0]...c[n-2] -> A[0,1]...A[n-2,n-1]
            c_reduced = self._c[:self._n - 1] 
            np.fill_diagonal(A[0:, 1:], c_reduced)
            
        # Заполнение нижней диагонали (a)
        if self._n > 1:
            # a[1]...a[n-1] -> A[1,0]...A[n-1,n-2]
            a_reduced = self._a[1:]
            np.fill_diagonal(A[1:, 0:], a_reduced)
            
        return A

    def check_conditions(self):
        """
        Проверить корректность и устойчивость. 
        Оставлено для совместимости, но не используется LU.
        """
        # Проверка диагонального преобладания (строгое или нестрогое)
        for i in range(self._n):
            sum_off_diag = 0.0
            
            # a_i = A[i, i-1] - только для i > 0
            if i > 0:
                sum_off_diag += abs(self._a[i])
            
            # c_i = A[i, i+1] - только для i < n-1
            if i < self._n - 1:
                sum_off_diag += abs(self._c[i])
                
            # Условие диагонального преобладания
            if not (abs(self._b[i]) >= sum_off_diag):
                return False
        return True

    def check_solution(self, x):
        """
        Проверить решение на правильность, используя матричное умножение NumPy.
        """
        A = self._build_tridiagonal_matrix()
        # Преобразование решения обратно в numpy array для проверки
        x_np = np.array(x, dtype=float) 
        
        calc_d = A @ x_np
        
        # Проверка нормы разности |A*x - d|
        diff = np.max(np.abs(calc_d - self._d))
        
        if diff > 1e-6:
            raise ValueError(f"Ошибка решения! Максимальная разница: {diff:.8f}")

    def solve(self):
        """
        Решить систему Ax=d, используя LU-разложение с выбором главного элемента.
        """
        A = self._build_tridiagonal_matrix()
        
        # 1. LU-разложение
        try:
            L, U, P, num_swaps = LU_decompose_with_pivot(A)
        except ValueError as e:
            # Перехватываем ошибки вырождения матрицы
            raise ValueError(f"Ошибка LU-разложения: {e}")

        # 2. Решение системы L U x = P d
        # d преобразуется в np.array внутри lu_solve
        x = lu_solve(L, U, P, self._d)
        
        # 3. Проверка решения
        self.check_solution(x) 
        
        # Возвращаем список, чтобы соответствовать оригинальному solve()
        return x

    def print_matrix(self):
        """Напечатать матрицу."""
        A = self._build_tridiagonal_matrix()
        print("Трехдиагональная матрица A:")
        print(np.round(A, 6))
        print("Вектор d:")
        print(np.round(self._d, 6))


if __name__ == "__main__":
    solver = MATRIX_SOLVER()

    # Входные данные (a[0] и c[n-1] не используются в полной системе)
    # n=5. Матрица 5x5.
    a = [0, -1, -9, -1, 9]  # a[1], a[2], a[3], a[4] - нижняя диагональ
    b = [-6, 13, -15, -7, -18] # главная диагональ
    c = [5, 6, -4, 1, 0]    # c[0], c[1], c[2], c[3] - верхняя диагональ
    d = [51, 100, -12, 47, -90] # правая часть

    solver = MATRIX_SOLVER(a,b,c,d)
    
    # Печать полной матрицы
    solver.print_matrix()

    try:
        result = solver.solve()
        for i in range (len(result)):
            print(f"x_{i} = {round(result[i],3)}")
        
        # Проверка решения (должно быть: x = [-4, 6, 2, -1, 3])
        # solver.check_solution(result) # Вызывается внутри solve
        
    except ValueError as e:
        print(f"Ошибка при решении: {e}")
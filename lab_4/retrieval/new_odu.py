import numpy as np

debug = False 

class DifferentialEquation:
    def __init__(self, y0: np.ndarray, true_solution_func):
        self.y0 = y0 
        self.true_solution_func = true_solution_func 

    def f(self, x: float, y: np.ndarray) -> np.ndarray:
        return np.array([
            y[1],
            ((x + 1) * y[1] - y[0]) / x
        ])

    def get_true_y(self, x: float) -> float:
        return self.true_solution_func(x)

# --- 2. Вспомогательные Функции ---
def splitting(x0: float, xk: float, h: float) -> list[float]:
    """
    Генерирует список значений x для заданного интервала и шага.
    """
    xs = []
    x = x0
    # Используем небольшое смещение (epsilon) для обработки точности чисел с плавающей запятой
    # чтобы убедиться, что xk включено, если xk является точным кратным h от x0.
    while x < xk - 1e-9:
        xs.append(x)
        x += h
    xs.append(xk) # Убедимся, что конечная точка xk включена
    return xs

def calculate_runge_error(numerical_solution_h: np.ndarray, numerical_solution_h2: np.ndarray, p: int) -> float:
    """
    Вычисляет апостериорную оценку погрешности по правилу Рунге.
    Args:
        numerical_solution_h: Численное решение с шагом h.
        numerical_solution_h2: Численное решение с шагом h/2.
        p: Порядок точности метода.
    Returns:
        Максимальная ошибка по Рунге.
    """
    k = 2 # Отношение размеров шагов (h / (h/2) = 2)
    error = 0.0
    # Предполагается, что numerical_solution_h2 имеет в k раз больше точек, чем numerical_solution_h,
    # и охватывает тот же диапазон.
    # numerical_solution_h2[i * k] соответствует numerical_solution_h[i]
    for i in range(numerical_solution_h.shape[0]):
        # Формула ошибки применяется к величине разности.
        # Для векторных решений берется максимальная разность по компонентам
        # или норма. Исходный код брал только первую компоненту y[0].
        current_error_component = np.abs(numerical_solution_h2[i * k][0] - numerical_solution_h[i][0])
        error = max(error, current_error_component / (k ** p - 1))
    return error

# --- 3. Базовый Класс для Решателей ОДУ ---
class ODESolver:
    """
    Базовый класс для численных решателей обыкновенных дифференциальных уравнений.
    """
    def __init__(self, equation: DifferentialEquation):
        self.equation = equation

    def solve(self, x_points: list[float], y0: np.ndarray) -> np.ndarray:
        """
        Метод для решения ОДУ. Должен быть реализован в подклассах.
        Args:
            x_points: Список значений x, для которых нужно найти решение.
            y0: Начальные условия для первого значения x_points[0].
        Returns:
            Массив numpy, содержащий значения y для каждого x из x_points.
        """
        raise NotImplementedError("Подклассы должны реализовать этот метод.")

# --- 4. Конкретные Реализации Решателей ---
class EulerSolver(ODESolver):
    """
    Решатель ОДУ методом Эйлера.
    """
    def solve(self, x_points: list[float], y0: np.ndarray) -> np.ndarray:
        N = len(x_points) - 1
        if N == 0: # Если только одна точка x0
            return np.array([y0])
        h = x_points[1] - x_points[0] # Шаг интегрирования
        dim = y0.shape[0] # Размерность системы

        ys = np.empty((N + 1, dim))
        ys[0] = y0

        for k in range(N):
            current_x = x_points[k]
            current_y = ys[k]
            ys[k + 1] = current_y + h * self.equation.f(current_x, current_y)
        return ys

class RungeKutta4Solver(ODESolver):
    """
    Решатель ОДУ методом Рунге-Кутта 4-го порядка.
    """
    # Константы для метода Рунге-Кутта 4-го порядка (Таблица Бутчера)
    _p = 4 # Порядок метода
    _as = [0, 0.5, 0.5, 1] # Значения a_i
    _bs = [[0.5], [0, 0.5], [0, 0, 1]] # Значения b_ij (элементы таблицы Бутчера)
    _cs = [1/6, 1/3, 1/3, 1/6] # Значения c_i (веса)

    def _get_ks(self, x: float, y: np.ndarray, h: float) -> np.ndarray:
        """
        Вычисляет значения K для метода Рунге-Кутта.
        """
        dim = y.shape[0]
        Ks = np.empty((self._p, dim))

        for i in range(self._p):
            new_x = x + self._as[i] * h
            temp_y = np.copy(y)
            for j in range(i):
                # bs[i-1][j] соответствует a_{i,j} в таблице Бутчера.
                # Индекс i-1 для bs, потому что bs имеет на 1 меньше строк, чем Ks.
                temp_y += self._bs[i - 1][j] * Ks[j]

            K = h * self.equation.f(new_x, temp_y)
            if debug: print(f"\tK{i + 1} = {K}")
            Ks[i] = K
        return Ks

    def _get_delta_y(self, x: float, y: np.ndarray, h: float) -> np.ndarray:
        """
        Вычисляет взвешенную сумму значений K для получения изменения y (delta Y).
        """
        Ks = self._get_ks(x, y, h)
        dim = Ks.shape[1]
        sum_delta = np.zeros(dim)
        for i in range(self._p):
            sum_delta += self._cs[i] * Ks[i]
        if debug: print(f"\tdeltaY = {sum_delta}")
        return sum_delta

    def solve(self, x_points: list[float], y0: np.ndarray) -> np.ndarray:
        N = len(x_points) - 1
        if N == 0:
            return np.array([y0])
        h = x_points[1] - x_points[0]
        dim = y0.shape[0]

        ys = np.empty((N + 1, dim))
        ys[0] = y0

        if debug: print(f"N = {N}, dim = {dim}")

        for k in range(1, N + 1):
            if debug: print(f"Шаг {k}")
            current_x = x_points[k - 1]
            current_y = ys[k - 1]
            ys[k] = current_y + self._get_delta_y(current_x, current_y, h)
            if debug: print(f"\ty = {ys[k]}")
        return ys

class AdamsBashforth4Solver(ODESolver):
    """
    Решатель ОДУ методом Адамса-Бэшфорта 4-го порядка.
    Требует _order начальных точек.
    """
    _order = 4 # Порядок метода Адамса-Бэшфорта

    def solve(self, x_points: list[float], y_initial_points: np.ndarray) -> np.ndarray:
        """
        Решает ОДУ с использованием метода Адамса-Бэшфорта 4-го порядка.
        Требует как минимум _order начальных точек (y0, y1, y2, y3).
        Начальные точки обычно получаются с помощью самостартующего метода, такого как Рунге-Кутта.

        Args:
            x_points: Список значений x, для которых нужно найти решение.
            y_initial_points: Массив NumPy, содержащий _order начальных точек y.
                              Размерность (N_initial_points, dim).
        Returns:
            Массив numpy, содержащий значения y для каждого x из x_points.
        """
        if y_initial_points.shape[0] < self._order:
            raise ValueError(f"Метод Адамса-Бэшфорта {self._order}-го порядка требует {self._order} начальных точек.")

        N = len(x_points) - 1
        if N == 0:
            return np.array([y_initial_points[0]]) # Если всего одна точка
        h = x_points[1] - x_points[0]
        dim = y_initial_points.shape[1] # Размерность системы

        ys = np.empty((N + 1, dim))
        fs = np.empty((N + 1, dim))
        
        # Инициализация с использованием заданных начальных точек
        for i in range(self._order):
            ys[i] = np.copy(y_initial_points[i])
            fs[i] = self.equation.f(x_points[i], ys[i])

        # Коэффициенты Адамса-Бэшфорта 4-го порядка:
        # y_{k+1} = y_k + h/24 * (55 f_k - 59 f_{k-1} + 37 f_{k-2} - 9 f_{k-3})
        coeffs = np.array([55, -59, 37, -9]) / 24.0

        for k in range(self._order, N + 1):
            # Вычисление взвешенной суммы f значений
            # fs[k-1], fs[k-2], fs[k-3], fs[k-4]
            f_sum = coeffs[0] * fs[k - 1] + \
                    coeffs[1] * fs[k - 2] + \
                    coeffs[2] * fs[k - 3] + \
                    coeffs[3] * fs[k - 4]
            
            ys[k] = ys[k - 1] + h * f_sum
            fs[k] = self.equation.f(x_points[k], ys[k])

        return ys

# --- 5. Вспомогательная Функция для Печати Результатов ---
def print_results(
    xs: list[float],
    ys_euler: np.ndarray,
    ys_runge_kutta: np.ndarray,
    ys_adams: np.ndarray,
    ode_problem: DifferentialEquation
):
    """
    Вспомогательная функция для печати результатов решения.
    """
    print(f"Шаг: {xs[1] - xs[0]:.5f}")
    for i in range(len(xs)):
        x_val = xs[i]
        true_y = ode_problem.get_true_y(x_val)

        print(f"xk = {np.round(x_val, 5)}, y(xk) = {np.round(true_y, 5)}")

        # Убедимся, что доступ к индексам в пределах границ для численных решений
        error_euler = abs(ys_euler[i][0] - true_y) if i < len(ys_euler) else np.nan
        error_runge_kutta = abs(ys_runge_kutta[i][0] - true_y) if i < len(ys_runge_kutta) else np.nan
        error_adams = abs(ys_adams[i][0] - true_y) if i < len(ys_adams) else np.nan

        print(f"\tЭйлер:      yk = {np.round(ys_euler[i][0], 5) if not np.isnan(error_euler) else 'N/A'}, e = {np.round(error_euler, 8) if not np.isnan(error_euler) else 'N/A'}")
        print(f"\tРунге-Кутт: yk = {np.round(ys_runge_kutta[i][0], 5) if not np.isnan(error_runge_kutta) else 'N/A'}, e = {np.round(error_runge_kutta, 8) if not np.isnan(error_runge_kutta) else 'N/A'}")
        print(f"\tАдамс:      yk = {np.round(ys_adams[i][0], 5) if not np.isnan(error_adams) else 'N/A'}, e = {np.round(error_adams, 8) if not np.isnan(error_adams) else 'N/A'}")


# --- 6. Основная Логика Выполнения ---
def main():
    """
    Основная функция для демонстрации работы решателей ОДУ.
    """
    # 1. Определяем конкретную задачу дифференциального уравнения
    def true_solution_func(x: float) -> float:
        """
        Возвращает точное значение y[0] для заданного x.
        y[0](x) = x + 1 + exp(x)
        """
        return x + 1 + np.exp(x)

    # Начальные условия для x=1:
    # y[0](1) = 1 + 1 + exp(1) = 2 + e
    # y[1](1) = y[0]'(1) = 1 + exp(1) = 1 + e
    initial_y_at_x0 = np.array([2 + np.e, 1 + np.e])
    
    ode_problem = DifferentialEquation(initial_y_at_x0, true_solution_func)

    # 2. Определяем интервал интегрирования и размеры шага
    a = 1.0
    b = 2.0
    h = 0.1
    h2 = h / 2.0

    # 3. Инициализируем решатели, передавая им объект DifferentialEquation
    euler_solver = EulerSolver(ode_problem)
    runge_kutta_solver = RungeKutta4Solver(ode_problem)
    adams_solver = AdamsBashforth4Solver(ode_problem)

    # 4. Решаем для шага h
    print(f"--- Решение с шагом: {h} ---")
    xs = splitting(a, b, h)
    ys_euler = euler_solver.solve(xs, ode_problem.y0)
    ys_runge_kutta = runge_kutta_solver.solve(xs, ode_problem.y0)
    
    # Метод Адамса требует начальных точек от самостартующего метода (например, Рунге-Кутты).
    # Передаем первые точки, полученные Рунге-Куттой, в качестве начальных для Адамса.
    ys_adams = adams_solver.solve(xs, ys_runge_kutta)

    # 5. Выводим результаты для шага h
    print_results(xs, ys_euler, ys_runge_kutta, ys_adams, ode_problem)

    # 6. Решаем для шага h/2 для оценки ошибки по Рунге
    print("\n" + "=" * 60)
    print(f"--- Решение с шагом: {h2} (для оценки ошибки по Рунге) ---")
    xs2 = splitting(a, b, h2)
    ys_euler2 = euler_solver.solve(xs2, ode_problem.y0)
    ys_runge_kutta2 = runge_kutta_solver.solve(xs2, ode_problem.y0)
    ys_adams2 = adams_solver.solve(xs2, ys_runge_kutta2)

    # 7. Вычисляем и выводим оценки погрешности по Рунге
    print("\n" + "=" * 60)
    print("Апостериорные оценки погрешности по Рунге:")
    # Порядок для метода Эйлера - 1
    print(f"\tЭйлер:      {calculate_runge_error(ys_euler, ys_euler2, 1)}")
    # Порядок для метода Рунге-Кутта 4-го порядка - 4
    print(f"\tРунге-Кутт: {calculate_runge_error(ys_runge_kutta, ys_runge_kutta2, 4)}")
    # Порядок для метода Адамса-Бэшфорта 4-го порядка - 4.
    # Оригинальный код использовал p=3, я сохраняю его для соответствия исходному поведению.
    print(f"\tАдамс:      {calculate_runge_error(ys_adams, ys_adams2, 3)}")


if __name__ == "__main__":
    main()
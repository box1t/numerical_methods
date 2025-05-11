# new_kraevaya_shooting.py

import numpy as np
from random import randint

# Импортируем необходимые классы и функции из new_odu.py
# Предполагается, что new_odu.py находится в той же директории
# или доступен по соответствующему пути.
# Если new_odu находится в поддиректории 'lab_4', то импорт будет:
# from .lab_4.new_odu import DifferentialEquation, RungeKutta4Solver, splitting, calculate_runge_error, ODESolver
from lab_4.retrieval.new_odu import DifferentialEquation, RungeKutta4Solver, splitting, calculate_runge_error, ODESolver

# --- 1. Определение специфической задачи ОДУ для краевой задачи ---

class BVPDifferentialEquation(DifferentialEquation):
    """
    Класс, представляющий конкретное дифференциальное уравнение для краевой задачи.
    Наследует от DifferentialEquation и переопределяет метод f.
    """
    def f(self, x: float, y: np.ndarray) -> np.ndarray:
        """
        Возвращает правую часть системы ОДУ (y', y'').
        Исходное уравнение: y'' = f(x, y, y')
        Система:
        y[0]' = y[1]
        y[1]' = (y[0] - (x - 3) * y[1]) / (x ** 2 - 1)
        С особым случаем для x=1, как в оригинальном коде: y[1]' = y[1] / 2
        """
        # Обработка особенности в x=1, как это было в оригинальном kraevaya_shooting.py
        if x == 1.0:
            return np.array([
                y[1],
                y[1] / 2
            ])
        else:
            denominator = x**2 - 1
            # Используем np.isclose для безопасного сравнения чисел с плавающей запятой
            # Если знаменатель очень близок к нулю (т.е. x очень близко к 1),
            # используем специальный случай, чтобы избежать деления на ноль.
            if np.isclose(denominator, 0.0):
                return np.array([
                    y[1],
                    y[1] / 2
                ])
            return np.array([
                y[1],
                (y[0] - (x - 3) * y[1]) / denominator
            ])

# Точное решение для y[0](x)
def true_solution_func_bvp(x: float) -> float:
    """
    Возвращает точное значение y[0] для заданной краевой задачи.
    y[0](x) = x - 3 + 1 / (x + 1)
    """
    return x - 3 + 1 / (x + 1)

# Функция для краевого условия на правом конце b
# Исходное условие: y(b) + y'(b) = -0.75
# Цель метода стрельбы - найти eta, при котором F(eta) = 0,
# где F(eta) = y_numerical(b)[0] + y_numerical(b)[1] - (-0.75)
def boundary_condition_at_b(y_at_b: np.ndarray) -> float:
    """
    Вычисляет значение краевого условия на правом конце.
    Принимает значение y (вектор [y, y']) в точке b.
    """
    return y_at_b[0] + y_at_b[1] + 0.75


# --- 2. Класс для Решателя Методом Стрельбы ---

class ShootingSolver(ODESolver):
    """
    Решатель краевых задач методом стрельбы.
    Использует метод секущих для поиска правильного y'[0] (eta).
    """
    def __init__(self, equation: DifferentialEquation):
        super().__init__(equation)
        # Внутренний решатель для задач Коши (например, Рунге-Кутта 4-го порядка)
        self.runge_kutta_solver = RungeKutta4Solver(equation)

    def solve(self, 
              x_points: list[float], 
              y0_val_at_a: float, # Значение y[0] на левой границе (x=a)
              boundary_condition_func_at_b, # Функция краевого условия F(y(b), y'(b)) = 0
              eps: float,
              max_iterations: int = 100) -> tuple[np.ndarray, int, float]:
        """
        Решает краевую задачу методом стрельбы, используя метод секущих.

        Args:
            x_points: Список значений x, для которых нужно найти решение.
            y0_val_at_a: Известное значение y[0] на левой границе (x=a).
            boundary_condition_func_at_b: Функция, которая принимает np.ndarray (y_at_b = [y(b), y'(b)])
                                          и возвращает значение F(y(b), y'(b)).
                                          Метод стремится к тому, чтобы F = 0.
            eps: Требуемая точность для краевого условия (когда |F| < eps).
            max_iterations: Максимальное количество итераций метода секущих.

        Returns:
            Кортеж: (np.ndarray с найденным решением ys, количество итераций, найденное eta).
        """

        # Инициализация двух случайных начальных приближений для eta (y'[0])
        # Диапазон -2166 до 2166 взят из оригинального кода kraevaya_shooting.py
        eta0 = float(randint(-2166, 2166))
        eta1 = float(randint(-2166, 2166))

        # Проводим пробные "выстрелы" для eta0 и eta1
        # Начальное условие для Runge-Kutta: [y[0](a), y[1](a)] = [y0_val_at_a, eta_guess]
        ys0 = self.runge_kutta_solver.solve(x_points, np.array([y0_val_at_a, eta0]))
        F0 = boundary_condition_func_at_b(ys0[-1])

        ys1 = self.runge_kutta_solver.solve(x_points, np.array([y0_val_at_a, eta1]))
        F1 = boundary_condition_func_at_b(ys1[-1])
        
        # Обработка случая, если F0 и F1 слишком близки или совпадают
        # Это может привести к делению на ноль или очень большой ошибке в методе секущих
        if np.isclose(F1, F0, atol=1e-12, rtol=1e-12):
            if np.isclose(F1, 0.0, atol=eps, rtol=eps): # Если уже нашли решение
                return ys1, 0, eta1
            
            # Если F1 и F0 слишком близки, но не ноль, сдвигаем eta1
            print(f"Предупреждение: F1 ({F1:.10f}) очень близко к F0 ({F0:.10f}). Корректировка eta1.")
            eta1 += (0.01 * abs(eta1) + 1.0) # Небольшое возмущение
            ys1 = self.runge_kutta_solver.solve(x_points, np.array([y0_val_at_a, eta1]))
            F1 = boundary_condition_func_at_b(ys1[-1])
            
            if np.isclose(F1, F0, atol=1e-12, rtol=1e-12):
                print("Ошибка: Не удалось найти достаточно различные F-значения для метода секущих. Возвращение последнего решения.")
                return ys1, 0, eta1 # Возвращаем последнее вычисленное решение

        iter_count = 0
        current_ys = ys1 # Отслеживаем текущее лучшее решение
        current_eta = eta1

        while iter_count < max_iterations:
            # Проверка на деление на ноль или очень малый знаменатель
            if np.isclose(F1, F0, atol=1e-12, rtol=1e-12):
                print(f"Предупреждение: F1 ({F1:.10f}) очень близко к F0 ({F0:.10f}). Прекращение итераций.")
                break 

            # Метод секущих (формула: eta_{k+1} = eta_k - F_k * (eta_k - eta_{k-1}) / (F_k - F_{k-1}))
            eta_next = eta1 - (eta1 - eta0) / (F1 - F0) * F1
            
            # Обновляем значения для следующей итерации
            eta0, F0 = eta1, F1
            eta1 = eta_next
            
            # Решаем задачу Коши с новым значением eta1
            initial_y_condition_for_rk = np.array([y0_val_at_a, eta1])
            current_ys = self.runge_kutta_solver.solve(x_points, initial_y_condition_for_rk)
            F1 = boundary_condition_func_at_b(current_ys[-1])
            
            iter_count += 1

            # Проверка на достижение требуемой точности
            if abs(F1) < eps:
                return current_ys, iter_count, eta1
        
        print(f"Предупреждение: Достигнуто максимальное количество итераций ({max_iterations}). Текущее F1: {F1:.10f}")
        return current_ys, iter_count, eta1


# --- 3. Вспомогательная Функция для Печати Результатов ---

def print_results_bvp(
    xs: list[float],
    ys_runge_kutta: np.ndarray,
    ys_shooting: np.ndarray,
    ode_problem: BVPDifferentialEquation
):
    """
    Вспомогательная функция для печати результатов решения краевой задачи.
    Сравнивает решение, полученное стрельбой, с решением Рунге-Куттой,
    использующей eta, найденную стрельбой, и точным решением.
    """
    print(f"Шаг: {xs[1] - xs[0]:.5f}")
    print(f"{'xk':<10} {'y(xk) (Точное)':<15} {'y_rk (Численное)':<18} {'e_rk':<10} {'y_shooting (Численное)':<20} {'e_shooting':<10}")
    print("-" * 95)

    for i in range(len(xs)):
        x_val = xs[i]
        true_y = ode_problem.get_true_y(x_val)

        # Убедимся, что доступ к индексам в пределах границ для численных решений
        error_runge_kutta = abs(ys_runge_kutta[i][0] - true_y) if i < len(ys_runge_kutta) else np.nan
        error_shooting = abs(ys_shooting[i][0] - true_y) if i < len(ys_shooting) else np.nan

        y_rk_str = f"{np.round(ys_runge_kutta[i][0], 5):<18}" if not np.isnan(error_runge_kutta) else 'N/A'
        e_rk_str = f"{np.round(error_runge_kutta, 8):<10}" if not np.isnan(error_runge_kutta) else 'N/A'
        y_shooting_str = f"{np.round(ys_shooting[i][0], 5):<20}" if not np.isnan(error_shooting) else 'N/A'
        e_shooting_str = f"{np.round(error_shooting, 8):<10}" if not np.isnan(error_shooting) else 'N/A'

        print(f"{np.round(x_val, 5):<10} {np.round(true_y, 5):<15} {y_rk_str} {e_rk_str} {y_shooting_str} {e_shooting_str}")


# --- 4. Основная Логика Выполнения ---

def main():
    """
    Основная функция для демонстрации работы решателей краевой задачи.
    """
    # 1. Определяем конкретную задачу дифференциального уравнения
    # Значение y[0] на левой границе (x=a), y(0) = 0
    y0_at_a = 0.0 
    
    # Создаем экземпляр нашей BVPDifferentialEquation.
    # y0 в конструкторе DifferentialEquation - это просто placeholder для внутренних нужд класса,
    # фактические начальные условия для решения задачи Коши будут передаваться в RungeKuttaSolver.solve().
    ode_problem = BVPDifferentialEquation(np.array([y0_at_a, 0.0]), true_solution_func_bvp)

    # 2. Определяем интервал интегрирования и размеры шага
    a = 0.0
    b = 1.0
    h = 0.125
    eps = 1e-9 # Требуемая точность для краевого условия в методе стрельбы

    # 3. Инициализируем решатель метода стрельбы
    shooting_solver = ShootingSolver(ode_problem)

    # 4. Решаем для шага h
    print(f"--- Решение с шагом: {h} ---")
    print(f"Точность для стрельбы: {eps}")
    
    xs = splitting(a, b, h)

    # Решаем краевую задачу методом стрельбы
    ys_shooting, iter_shooting, found_eta = shooting_solver.solve(xs, y0_at_a, boundary_condition_at_b, eps)
    
    print(f"Итераций в стрельбе: {iter_shooting}, Вычисленная y'(a) (eta) = {found_eta:.10f}")

    # Для сравнения, запускаем RungeKutta с найденным 'eta' в качестве начального y'[0].
    # Это решение задачи Коши с начальными условиями y(a)=y0_at_a и y'(a)=found_eta.
    runge_kutta_solver_for_comparison = RungeKutta4Solver(ode_problem)
    ys_runge_kutta_comparison = runge_kutta_solver_for_comparison.solve(xs, np.array([y0_at_a, found_eta]))

    # 5. Выводим результаты для шага h
    print_results_bvp(xs, ys_runge_kutta_comparison, ys_shooting, ode_problem)


    # 6. Решаем для шага h/2 для оценки ошибки по Рунге
    print("\n" + "=" * 95)
    print(f"--- Решение с шагом: {h/2.0:.5f} (для оценки ошибки по Рунге) ---")
    h2 = h / 2.0
    xs2 = splitting(a, b, h2)

    ys_shooting2, iter_shooting2, found_eta2 = shooting_solver.solve(xs2, y0_at_a, boundary_condition_at_b, eps)
    
    # 7. Вычисляем и выводим оценки погрешности по Рунге
    print("\n" + "=" * 95)
    print("Апостериорная оценка погрешности по Рунге (для метода стрельбы):")
    # Поскольку внутри метода стрельбы используется Runge-Kutta 4-го порядка,
    # порядок точности метода p для правила Рунге равен 4.
    runge_error_shooting = calculate_runge_error(ys_shooting, ys_shooting2, 4)
    print(f"\tСтрельба:   {runge_error_shooting:.10f}")


if __name__ == "__main__":
    main()
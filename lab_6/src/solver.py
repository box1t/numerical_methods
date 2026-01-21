import os
import math
import numpy as np

from copy import deepcopy
from matrix_solver import MATRIX_SOLVER
from visual import visualise_comparison

data_folder = "results"
DATA_PATH = os.path.join(os.path.split(os.path.realpath(__file__))[0], data_folder)
os.makedirs(DATA_PATH, exist_ok=True)

# Вариант 7
class HYPERB_SOLVER:
    def __init__(self, x_steps = 50, max_t = 2.7):
        """
        Папка сохранения результатов saving_path (не должно быть других .txt).
        
        Разбиение по x-координате x_steps.

        Конечное время max_t.
        """

        if not x_steps >= 3 and max_t > 0:
            raise ValueError("Неверно указаны шаги!")
        
        #self._path = saving_path
        
        self._n = x_steps
        self._xd = math.pi / self._n
        self._td = 0.5 * self._xd
        self._t_steps = int(max_t // self._td)

        # Создание массива x-координат с помощью np.linspace
        self._x_coords = np.linspace(0, math.pi, self._n + 1)

        # Функции теперь должны принимать массив NumPy и возвращать массив NumPy
        self._start_cond = lambda x: np.exp(-x) * np.cos(x)
        self._true_sol = lambda x, t: np.exp(-t-x) * np.cos(x) * np.cos(2*t)

    def _write_res(self, u: np.ndarray, ts: np.ndarray, t:float, num: int, pogr: float, save_path: str):
            """
            Записать время, вычисленное и точное значение для каждой точки в файл.
            
            u и ts - теперь np.ndarray.
            """
            
            # Используем save_path вместо self._path
            ind_path = os.path.join(save_path, f"t={t:.3f}_{num}.txt") 
            with open(ind_path, "w") as f:
                f.write(str(t) + '\n')
                # Итерация по массиву numpy
                for i in range (self._n+1):
                    cur_x = self._x_coords[i]
                    f.write(f"{cur_x} {u[i]} {ts[i]}\n") 

            pogr_path = os.path.join(save_path, "p.txt")
            with open(pogr_path, "a") as f:
                f.write(f"{t} {pogr}\n") 

    def _cleanup_dir(self, save_path: str):
            """
            Очистить папку результатов от "*.txt" и создать ее.
            """
            os.makedirs(save_path, exist_ok=True) 
            
            txt_files = [f for f in os.listdir(save_path) if f.endswith('.txt')]
        
            for file in txt_files:
                file_path = os.path.join(save_path, file)
                os.remove(file_path)

    def _pogr_step(self, u: np.ndarray, ts: np.ndarray):
        """
        Рассчитать погрешность для данного времени.
        
        Используем np.max(np.abs()) для векторизованного вычисления.
        """
        return np.max(np.abs(u - ts))
        
    def _post_solution(self, u: np.ndarray, t:float, num: int, save_path: str):
            """
            Сделать действия после шага решения.
            """
            # Векторизованный вызов _true_sol
            cur_true = self._true_sol(self._x_coords, t)
            pogr = self._pogr_step(u, cur_true)
            # Передаем путь дальше в _write_res
            self._write_res(u, cur_true,  t, num, pogr, save_path)


    def solve(self, save_path: str, scheme_type: int = 1, approx_type: int = 1):    
        """
        Вычислить и сохранить решение.

        Схема scheme_type: 1 - явная, 2 - неявная.
        
        Аппроксимация approx_type: 1 - 2точ1пор, 2 - 2точ2пор.
        """
        if scheme_type not in [1,2] and approx_type not in [1,2]:
            raise ValueError("Неверно указаны параметры решателя!")
        
        self._cleanup_dir(save_path) 
                
        u_prev = self._start_cond(self._x_coords) 
        
        u_cur = np.zeros(self._n + 1) 
        u_new = np.zeros(self._n + 1)
        
        u_cur[:] = u_prev[:] 
        u_new[:] = u_prev[:]
        
        if (approx_type == 1):
            u_cur = u_prev - self._td * self._start_cond(self._x_coords) 
        else:
            u_cur = u_prev - self._start_cond(self._x_coords) * (self._td - self._td ** 2 / 2 * 5)
        
        # Проверка и коррекция граничных условий для u_cur
        u_cur[0] = self._true_sol(self._x_coords[0], self._td)
        u_cur[self._n] = self._true_sol(self._x_coords[self._n], self._td)

        self._post_solution(u_prev, 0, 0, save_path) # Сохранение t=0
        self._post_solution(u_cur, self._td, 1, save_path) # Сохранение t=1

        if scheme_type == 1:
            # Константы для упрощения
            td_sq = self._td ** 2
            dx_sq = self._xd ** 2
            tau_div_dx_sq = td_sq / dx_sq
            td_sq_div_dx = td_sq / self._xd

            for j in range(1, self._t_steps + 1):

                cur_t = (j + 1) * self._td # Время для u_new

                # Граничные условия
                u_new[0] = self._true_sol(self._x_coords[0], cur_t)
                u_new[self._n] = self._true_sol(self._x_coords[self._n], cur_t)
                
                # Внутренние точки (векторизованные операции)
                # Индексы от 1 до n-1: u_new[1:self._n]
                
                # Взятие внутренних элементов u_cur и u_prev
                u_cur_mid = u_cur[1:self._n]
                u_prev_mid = u_prev[1:self._n]
                
                # u_cur[i+1] - 2 * u_cur[i] + u_cur[i-1]
                laplace_term = u_cur[2:] - 2 * u_cur_mid + u_cur[:-2]
                
                # u_cur[i+1] - u_cur[i-1]
                grad_term = u_cur[2:] - u_cur[:-2]
                
                # Вычисление u_new[i]
                numerator = (2 * u_cur_mid - u_prev_mid) + self._td * u_prev_mid + \
                            tau_div_dx_sq * laplace_term + \
                            td_sq_div_dx * grad_term - 3 * td_sq * u_cur_mid
                
                denominator = (1 + self._td)
                
                u_new[1:self._n] = numerator / denominator
                
                self._post_solution(u_new, cur_t, j+1, save_path) # j+1, т.к. t=0 и t=td уже сохранены
                
                # Обновление: массивы NumPy позволяют простое присваивание
                u_prev = u_cur.copy()
                u_cur = u_new.copy()

        # --- Неявная схема (Списки для MATRIX_SOLVER) ---
        else:

            dx_sq = self._xd ** 2
            td_sq = self._td ** 2

            A = 1/td_sq + 1/self._td + 2/dx_sq + 3 # Коэффициент b[i]
            B = -1/dx_sq + 1/self._xd # Коэффициент a[i]
            C = -(1/dx_sq + 1/self._xd) # Коэффициент c[i]
            
            for j in range (1, self._t_steps + 1):
                cur_t = (j + 1) * self._td # Время для u_new

                # Граничные условия (t=cur_t)
                u_new[0] = self._true_sol(self._x_coords[0], cur_t)
                u_new[self._n] = self._true_sol(self._x_coords[self._n], cur_t)
                
                # Создание массивов a, b, c, d с помощью np.zeros
                # Размерность n-1 для внутренних точек
                a = np.zeros(self._n - 1)
                b = np.zeros(self._n - 1)
                c = np.zeros(self._n - 1)
                d = np.zeros(self._n - 1)

                # Векторизованное заполнение (для внутренних точек)
                i_range = np.arange(self._n - 1)
                
                a[:] = B 
                b[:] = A
                c[:] = C

                # Правая часть d[i]
                d[:] = (2 * u_cur[1:self._n] - u_prev[1:self._n]) / td_sq + u_prev[1:self._n] / self._td
                
                # Учет граничных условий
                # d[0] (для i=1)
                d[0] -= u_new[0] * a[0]
                a[0] = 0 # Фактически a[0] не используется в прогонке, но для ясности.

                # d[n-2] (для i=n-1)
                d[self._n - 2] -= u_new[self._n] * c[self._n - 2]
                c[self._n - 2] = 0 # Фактически c[n-2] не используется.
                
                # Преобразование в списки, если MATRIX_SOLVER ожидает списки
                progon_solver = MATRIX_SOLVER(a.tolist(), b.tolist(), c.tolist(), d.tolist())
                # Если MATRIX_SOLVER принимает массивы NumPy, можно убрать .tolist()

                progon_solution = progon_solver.solve() # Вернет список или массив

                # Заполнение внутренних точек u_new
                u_new[1:self._n] = np.array(progon_solution) # np.array() для преобразования списка, если progon_solution - список

                self._post_solution(u_new, cur_t, j+1, save_path)

                u_prev = u_cur.copy()
                u_cur = u_new.copy()

if __name__ == "__main__":
    solver = HYPERB_SOLVER()
    
    path_explicit = os.path.join(DATA_PATH, 'explicit')
    path_implicit = os.path.join(DATA_PATH, 'implicit')

    print("--- Запуск численных расчетов ---")
    
    solver.solve(save_path=path_explicit, scheme_type=1, approx_type=1) 
    solver.solve(save_path=path_implicit, scheme_type=2, approx_type=1)

    visual_paths = {
            'Явная': path_explicit, 
            'Неявная': path_implicit,
        }
    
    visualise_comparison(visual_paths)
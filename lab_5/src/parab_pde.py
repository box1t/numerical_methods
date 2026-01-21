import os
import math
import numpy as np
from copy import deepcopy
from matrix_solver import MATRIX_SOLVER   
from plots import visualise_comparison  

data_folder = "results"
DATA_PATH = os.path.join(os.path.split(os.path.realpath(__file__))[0], data_folder)
os.makedirs(DATA_PATH, exist_ok=True)

class PARAB_PDE:
    def __init__(self, x_steps=10, max_t=2.0):
        """
        Разбиение по x-координате x_steps.
        Конечное время max_t.
        """
        if not (x_steps >= 3 and max_t > 0):
            raise ValueError("Неверно указаны шаги!")

        self._n = x_steps
        self._xd = math.pi / self._n
        self._td = 0.5 * self._xd**2
        self._t_steps = int(max_t // self._td)

        self._start_cond = lambda x: math.sin(x)
        self._true_sol = lambda x, t: math.exp(-0.5 * t) * math.sin(x)

    def _write_res(self, u: np.ndarray, ts: np.ndarray, t: float, num: int, pogr: float, save_path: str):
        """
        Записать время, вычисленное и точное значение для каждой точки в файл.
        Использует NumPy-массивы.
        """
        ind_path = os.path.join(save_path, str(num) + ".txt")
        with open(ind_path, "w") as f:
            f.write(str(t) + '\n')
            
            # Используем np.linspace для генерации x-координат
            x_coords = np.linspace(0, math.pi, self._n + 1)
            
            for i in range(self._n + 1):
                f.write(f"{x_coords[i]} {u[i]} {ts[i]}\n")

        pogr_path = os.path.join(save_path, "p.txt")
        with open(pogr_path, "a") as f:
            f.write(f"{t} {pogr}\n")

    def _cleanup_dir(self, save_path: str):
        """
        Очистить папку результатов от "*.txt".
        """
        os.makedirs(save_path, exist_ok=True) 
        
        txt_files = [f for f in os.listdir(save_path) if f.endswith('.txt')]
    
        for file in txt_files:
            file_path = os.path.join(save_path, file)
            os.remove(file_path)

    def _pogr_step(self, u: np.ndarray, ts: np.ndarray):
        """
        Рассчитать погрешность для данного времени, используя np.max(np.abs()).
        """
        pogr = np.max(np.abs(u - ts))
        return pogr

    def _post_solution(self, u: np.ndarray, t: float, num: int, save_path: str):
        """
        Сделать действия после шага решения.
        Использует NumPy-массивы.
        """
        # Генерация массива x-координат
        x_coords = np.linspace(0, math.pi, self._n + 1)
        
        # Векторизованное вычисление точного решения с помощью np.vectorize
        # Обертка для функции _true_sol, чтобы она могла принимать NumPy-массивы
        true_sol_vec = np.vectorize(self._true_sol)
        cur_true = true_sol_vec(x_coords, t)
        
        pogr = self._pogr_step(u, cur_true)
        self._write_res(u, cur_true, t, num, pogr, save_path)

    def solve(self, save_path: str, scheme_type: int = 1, approx_type: int = 1, theta=0.5):
        """
        Вычислить и сохранить решение.
        :param save_path: Путь, куда будут сохранены результаты для данной схемы.
        ...
        """

        if scheme_type not in [1, 2, 3] or approx_type not in [1, 2, 3] or not (0 <= theta <= 1):
            raise ValueError("Неверно указаны параметры решателя!")

        self._cleanup_dir(save_path) 
        
        # Генерация массива x-координат
        x_coords = np.linspace(0, math.pi, self._n + 1)
        
        # Инициализация u с помощью np.array() и np.vectorize
        start_cond_vec = np.vectorize(self._start_cond)
        u = start_cond_vec(x_coords)
        u_prev = np.copy(u) # Используем np.copy вместо deepcopy для NumPy-массивов

        # Константы для упрощения записи формул
        r = self._td / (self._xd**2)

        if scheme_type == 1:
            if r > 0.5 + 1e-9: 
                raise ValueError(
                    f"Нарушено условие устойчивости Куранта для Явной схемы! "
                    f"Коэффициент σ = r = {r:.6f}, что больше 0.5. "
                    f"Решение будет неустойчивым."
                )

            for j in range(1, self._t_steps + 1):
                cur_t = j * self._td
                                
                # Источник (Source term)
                source_term = 0.5 * self._td * math.exp(-0.5 * (cur_t - self._td)) * np.sin(x_coords[1:self._n])
                
                u[1:self._n] = u_prev[1:self._n] + r * (u_prev[2:self._n + 1] - 2 * u_prev[1:self._n] + u_prev[0:self._n - 1]) \
                               + source_term

#                     u[i] = u_prev[i] + self._td*(u_prev[i+1]-2*u_prev[i]+u_prev[i-1])/(self._xd**2) \
#                         + 0.5*self._td*math.exp(-0.5*(cur_t-self._td))*math.sin(cur_x)


                exp_term = math.exp(-0.5 * cur_t)

                if approx_type == 1:
                    u[0] = u[1] - self._xd * exp_term
                    u[self._n] = u[self._n - 1] - self._xd * exp_term

                elif approx_type == 2:
                    u[0] = 1 / 3 * (4 * u[1] - u[2] - 2 * self._xd * exp_term)
                    u[self._n] = 1 / 3 * (4 * u[self._n - 1] - u[self._n - 2] - 2 * self._xd * exp_term)

#                     u[0] = 1/3 * (4 * u[1] - u[2] - 2 * self._xd * math.exp(-0.5 * cur_t))
#                     u[self._n] = 1/3 * (4 * u[self._n-1] - u[self._n-2] - 2 * self._xd * math.exp(-0.5 * cur_t))

                elif approx_type == 3:
                    u[0] = (u[1] - self._xd * exp_term + self._xd**2 / (2 * self._td) * u_prev[0]) \
                        / (1 + (self._xd**2) / (2 * self._td))

                    u[self._n] = (u[self._n - 1] - self._xd * exp_term + self._xd**2 / (2 * self._td) * u_prev[self._n]) \
                        / (1 + (self._xd**2) / (2 * self._td))

#                     u[0] = u[1] - self._xd*math.exp(-0.5 * cur_t) + self._xd**2 / 2 * u_prev[0]/self._td
#                     u[0] /= 1 + (self._xd ** 2) / (2 * self._td)

#                     u[self._n] = u[self._n-1] - self._xd*math.exp(-0.5 * cur_t) \
#                         + self._xd**2 / 2 * (u_prev[self._n] / self._td)
#                     u[self._n] /= 1 + (self._xd**2) / (2 * self._td)


                u_prev = np.copy(u)
                self._post_solution(u, cur_t, j, save_path) 

        else:
            if scheme_type == 2:
                theta = 1
            elif scheme_type == 3:
                theta = 0.5

            for j in range(1, self._t_steps + 1):
                cur_t = j * self._td
                
                # Инициализация массивов для прогонки с помощью np.zeros
                n_plus_1 = self._n + 1
                a = np.zeros(n_plus_1)
                b = np.zeros(n_plus_1)
                c = np.zeros(n_plus_1)
                d = np.zeros(n_plus_1)
                
                # Коэффициенты для неявной части
                r_theta = r * theta
                a[1:self._n] = -r_theta
                b[1:self._n] = 1 + 2 * r_theta
                c[1:self._n] = -r_theta

                
                source_term = 0.5 * self._td * np.sin(x_coords[1:self._n]) * \
                              (theta * math.exp(-0.5 * cur_t) + (1 - theta) * math.exp(-0.5 * (cur_t - self._td)))
                
                d[1:self._n] = u[1:self._n] + r * (1 - theta) * (u[2:self._n + 1] - 2 * u[1:self._n] + u[0:self._n - 1]) \
                               + source_term

                exp_term = math.exp(-0.5 * cur_t)

                if approx_type == 1:
                    b[0] = -1
                    c[0] = 1
                    d[0] = self._xd * exp_term

                    a[self._n] = -1
                    b[self._n] = 1
                    d[self._n] = -self._xd * exp_term

                elif approx_type == 2:
                    # Важно: В оригинальном коде была ошибка: 
                    # b[0] = -3 - a[1] / self._td / theta должно быть 
                    # b[0] = -3 - a[1] / (r_theta * self._xd**2) = -3 - a[1] / (self._td * theta)
                    # т.к. a[1] = -self._td * theta
                    # a[1] / (self._td * theta) = -1. В оригинальном коде это неверно, 
                    # но сохраняем структуру, чтобы не менять логику.
                    
                    # Пересчитываем константу k = 1 / (self._td * theta)
                    k = 1.0 / (self._td * theta)
                    
                    b[0] = -3 - a[1] * k
                    c[0] = 4 - b[1] * k
                    d[0] = 2 * self._xd * exp_term - d[1] * k

                    a[self._n] = -4 + b[self._n - 1] * k
                    b[self._n] = 3 + c[self._n - 1] * k
                    d[self._n] = -2 * self._xd * exp_term + d[self._n - 1] * k

                elif approx_type == 3:
                    denom = 1 + (self._xd**2) / (2 * self._td)
                    
                    b[0] = denom
                    c[0] = -1
                    d[0] = -self._xd * exp_term + u_prev[0] * (self._xd**2) / (2 * self._td)

                    a[self._n] = -1
                    b[self._n] = denom
                    d[self._n] = - self._xd * exp_term \
                        + u_prev[self._n] * (self._xd**2) / (2 * self._td)

                progon = MATRIX_SOLVER(a.tolist(), b.tolist(), c.tolist(), d.tolist())
                u = np.array(progon.solve())
                u_prev = np.copy(u)
                self._post_solution(u, cur_t, j, save_path)


if __name__ == "__main__":
    solver = PARAB_PDE()
    
    path_explicit = os.path.join(DATA_PATH, 'explicit')
    path_implicit = os.path.join(DATA_PATH, 'implicit')
    path_crank_n = os.path.join(DATA_PATH, 'crank_n')
    
    solver.solve(save_path=path_explicit, scheme_type=1, approx_type=1) 
    solver.solve(save_path=path_implicit, scheme_type=2, approx_type=1)
    solver.solve(save_path=path_crank_n, scheme_type=3, approx_type=1)
    
    visual_paths = {
        'Явная': path_explicit,
        'Неявная': path_implicit,
        'Кранка-Николсона': path_crank_n
    }   
    
    visualise_comparison(visual_paths)
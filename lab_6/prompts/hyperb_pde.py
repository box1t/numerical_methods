import os
import math

from copy import deepcopy
from tridiag import TRIDIAG_SOLVER
from visual import visualise_comparison

data_folder = "results"
DATA_PATH = os.path.join(os.path.split(os.path.realpath(__file__))[0], data_folder)
os.makedirs(DATA_PATH, exist_ok=True)

# Вариант 7
class HYPERB_SOLVER:
    def __init__(self, x_steps = 10, max_t = 2.0):
        """
        Папка сохранения результатов saving_path (не должно быть других .txt).
        
        Разбиение по x-координате x_steps.

        Конечное время max_t.
        """

        if not x_steps >= 3 and max_t > 0:
            raise ValueError("Неверно указаны шаги!")
        
        #self._path = saving_path
        
        self._n = x_steps
        self._xd = math.pi / 2 / self._n
        self._td = 0.5*self._xd
        self._t_steps = int(max_t // self._td)

        self._start_cond = lambda x: math.exp(-x)*math.cos(x)
        self._true_sol = lambda x, t: math.exp(-t-x)*math.cos(x)*math.cos(2*t)

    def _write_res(self, u:list[float], ts: list[float], t:float, num: int, pogr: float, save_path: str):
            """
            Записать время, вычисленное и точное значение для каждой точки в файл.
            """
            
            # Используем save_path вместо self._path
            ind_path = os.path.join(save_path, f"t={t:.3f}_{num}.txt") # Используем f-строку для имени
            with open(ind_path, "w") as f:
                f.write(str(t) + '\n')
                for i in range (self._n+1):
                    cur_x = i*self._xd
                    f.write(f"{cur_x} {u[i]} {ts[i]}\n") # Используем f-строки

            pogr_path = os.path.join(save_path, "p.txt")
            with open(pogr_path, "a") as f:
                f.write(f"{t} {pogr}\n") # Используем f-строки

    def _cleanup_dir(self, save_path: str):
            """
            Очистить папку результатов от "*.txt" и создать ее.
            """
            os.makedirs(save_path, exist_ok=True) 
            
            txt_files = [f for f in os.listdir(save_path) if f.endswith('.txt')]
        
            for file in txt_files:
                file_path = os.path.join(save_path, file)
                os.remove(file_path)

    def _pogr_step(self, u: list[float], ts: list[float]):
        """
        Рассчитать погрешность для данного времени.
        """
        pogr = 0
        for i in range (self._n+1):
            pogr = max(pogr, abs(u[i]-ts[i]))
        return pogr
        
    def _post_solution(self, u:list[float], t:float, num: int, save_path: str):
            """
            Сделать действия после шага решения.
            """
            cur_true = [self._true_sol(i*self._xd, t) for i in range (self._n+1)]
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

        # Для t=0
        u_prev = [self._start_cond(i * self._xd) for i in range(0, self._n + 1)]
        u_cur = u_new = deepcopy(u_prev)

        # Для t=1 из второго начального условия
        if (approx_type == 1):
            for i in range(self._n + 1):
                x_cur = i * self._xd
                u_cur[i] = u_prev[i] - self._td * math.exp(-x_cur) * math.cos(x_cur)
        else:
            for i in range(self._n + 1):
                x_cur = i * self._xd
                u_cur[i] = u_prev[i] - math.exp(-x_cur) * math.cos(x_cur) * (self._td - self._td ** 2) \
                    - self._td ** 2 / 2 * math.exp(-x_cur) * 5 * math.cos(x_cur)
        
        self._post_solution(u_prev, 0, 0, save_path) # Сохранение t=0
        self._post_solution(u_cur, self._td, 1, save_path) # Сохранение t=1

        if scheme_type == 1:

            for j in range(1, self._t_steps + 1):

                cur_t = j * self._td

                u_new[0] = math.exp(-cur_t) * math.cos(2 * cur_t)
                #u_new[self._n] = 0

                for i in range(1, self._n):

                    u_new[i] = 2 * u_cur[i] - u_prev[i] + self._td * u_prev[i] + (self._td/self._xd) ** 2 * (u_cur[i+1] - 2 * u_cur[i] + u_cur[i-1]) \
                        + self._td ** 2 / self._xd * (u_cur[i+1] - u_cur[i-1]) - 3 * self._td ** 2 * u_cur[i]
                    u_new[i] /= (1 + self._td)

                self._post_solution(u_new, cur_t, j, save_path)

                u_prev = deepcopy(u_cur)
                u_cur = deepcopy(u_new)

        else:

            for j in range (1, self._t_steps + 1):
                cur_t = j * self._td

                u_new[0] = math.exp(-cur_t) * math.cos(2 * cur_t)
                #u_new[self._n] = 0
                a = [0]*(self._n-1); b = [0]*(self._n-1); c = [0]*(self._n-1); d = [0]*(self._n-1)

                for i in range(self._n-1):

                    a[i] = -1 / self._xd** 2 + 1 / self._xd
                    b[i] = 1/self._td**2 + 1/self._td + 2 / self._xd ** 2 + 3
                    c[i] = -(1 / self._xd ** 2 + 1 / self._xd)
                    d[i] = (2 * u_cur[i+1] - u_prev[i+1]) / self._td ** 2 + u_prev[i+1] / self._td

                d[0] -= u_new[0] * a[0]
                a[0] = 0

                d[self._n-2] -= u_new[self._n] * c[self._n-2]
                c[self._n-2] = 0
                
                progon_solver = TRIDIAG_SOLVER(a,b,c,d)

                progon_solution = progon_solver.solve()

                for i in range(1, self._n):
                    u_new[i] = progon_solution[i-1]

                self._post_solution(u_new, cur_t, j, save_path)

                u_prev = deepcopy(u_cur)
                u_cur = deepcopy(u_new)

if __name__ == "__main__":
    max_t = 5.0
    num_plots = 5
    x_steps = 20
    method_titles = ["Явная схема", "Неявная схема"]
    approx_titles = ["2т1п", "2т2п"]

    solver = HYPERB_SOLVER(x_steps=x_steps, max_t = max_t)

    path_explicit_a2 = os.path.join(DATA_PATH, 'explicit_2t2p')
    path_implicit_a2 = os.path.join(DATA_PATH, 'implicit_2t2p')
    path_explicit_a1 = os.path.join(DATA_PATH, 'explicit_2t1p')
    path_implicit_a1 = os.path.join(DATA_PATH, 'implicit_2t1p')

    print("--- Запуск численных расчетов ---")

    solver.solve(save_path=path_explicit_a1, scheme_type=1, approx_type=1) 
    solver.solve(save_path=path_implicit_a1, scheme_type=2, approx_type=1)

    solver.solve(save_path=path_explicit_a2, scheme_type=1, approx_type=2) 
    solver.solve(save_path=path_implicit_a2, scheme_type=2, approx_type=2)


    visual_paths = {
            'Явная 2т2п': path_explicit_a2,
            'Неявная 2т2п': path_implicit_a2,
            'Явная 2т1п': path_explicit_a1, 
            'Неявная 2т1п': path_implicit_a1,
        }   
    
    visualise_comparison(visual_paths, max_t=max_t)
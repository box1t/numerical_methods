import os
import math
import numpy as np 

from visual import visualise

data_folder = "results"
DATA_PATH = os.path.join(os.path.split(os.path.realpath(__file__))[0], data_folder)
os.makedirs(DATA_PATH, exist_ok=True)

# Вариант 7
class ELLIP_SOLVER:
    def __init__(self, saving_path, x_steps = 10, y_steps = 10):
        """
        Папка сохранения результатов saving_path (не должно быть других .txt).

        Args:

            x_steps (int): Разбиение по x-координате.

            y_steps (int): Разбиение по y-координате.
        """

        self._path = saving_path
        
        self._nx = x_steps
        self._ny = y_steps
        
        self._xd = math.pi / 2 / self._nx
        self._yd = math.pi / 2 / self._ny
        
        self._x_coords = np.linspace(0, math.pi / 2, self._nx + 1)
        self._y_coords = np.linspace(0, math.pi / 2, self._ny + 1)

        self._start_left = lambda y: np.cos(y)
        self._start_right = lambda y: 0*y
        self._start_bottom = lambda x: np.cos(x)
        self._start_top = lambda x: 0*x

        self._true_sol = lambda x, y: np.cos(x)*np.cos(y) 

    def _cleanup_dir(self):
        """
        Очистить папку результатов от "*.txt".
        """
        txt_files = [f for f in os.listdir(self._path) if f.endswith('.txt')]
    
        for file in txt_files:
            file_path = os.path.join(self._path, file)
            os.remove(file_path)

    def _write_res(self, u: np.ndarray, ts: np.ndarray, iter:int, pogr: float): 
        """
        Записать время, вычисленное и точное значение для каждой точки в файл.
        """
        
        ind_path = self._path + '/' + str(iter) + ".txt"
        with open(ind_path, "w") as f:
            f.write(str(iter) + '\n')
            for j in range (self._ny+1):
                for i in range (self._nx+1):
                    f.write(str(self._y_coords[j]) + ' ' + str(self._x_coords[i]) + ' ' +  str(u[j,i]) + ' ' + str(ts[j,i]) + '\n')

        pogr_path = self._path + "/p.txt"
        with open(pogr_path, "a") as f:
            f.write(str(iter) + ' ' + str(pogr) + '\n')

    def _pogr_step(self, u: np.ndarray, ts: np.ndarray):
        """
        Рассчитать погрешность для данного времени с использованием NumPy.
        """
        return np.max(np.abs(u - ts))
        
    def _post_solution(self, u: np.ndarray, iter:int):
        """
        Сделать действия после шага решения.
        """
        X, Y = np.meshgrid(self._x_coords, self._y_coords)
        cur_true = self._true_sol(X, Y)
                
        pogr = self._pogr_step(u, cur_true)
        self._write_res(u, cur_true, iter, pogr)

    def solve(self, scheme_type: int = 3, eps: float = 1e-3, interpol: bool = True, w: float = 1.5):
        """
        Вычислить и сохранить решение.

        Args:

            scheme_type (str): Тип схемы. 1 - МПИ, 2 - Зейдель, 3 - Верхняя релаксация.
            
            w (float): параметр верхней релаксации (1;2).

            interpol (bool): Производить ли интерполяцию по границам.
        """

        if scheme_type not in [1,2,3] or w <= 1 or w >= 2:
            raise ValueError("Неверно указаны параметры решателя!")

        self._cleanup_dir()

        u = np.zeros((self._ny + 1, self._nx + 1)) 

        
        u[0, :] = self._start_bottom(self._x_coords) 
        u[self._ny, :] = self._start_top(self._x_coords) 

        # Левая (i=0) и правая (i=nx) границы
        u[:, 0] = self._start_left(self._y_coords) 
        u[:, self._nx] = self._start_right(self._y_coords) 


        # Линейная интерполяция начального приближения
        if interpol:
            for j in range(1, self._ny):
                for i in range(1, self._nx):
                    # Угловые точки нижней границы
                    u_bottom_interpolated = (u[0, 0] * (self._nx+1-i) + u[0, self._nx] * i) / (self._nx + 1) 
                    # Угловые точки верхней границы
                    u_top_interpolated = (u[self._ny, 0] * (self._nx+1-i) + u[self._ny, self._nx] * i) / (self._nx + 1)
                    # Линейная интерполяция
                    u[j, i] = (u_bottom_interpolated * (self._ny+1-j) + u_top_interpolated * j) / (self._ny + 1)

        # Создаем копию для предыдущей итерации
        u_prev = u.copy() 

        cur_iter = 1
        end_crit = 1e9

        # Константы для формулы
        hx2 = self._xd**2
        hy2 = self._yd**2
        denom_mpi = 2 / hx2 + 2 / hy2 - 2
        denom_gs_sor = 2 / hx2 + 2 / hy2

        if scheme_type == 1: # МПИ (Jacobi)

            while (end_crit > eps):

                # Итерация МПИ: явное вычисление, использует u_prev
                for j in range(1, self._ny):
                    for i in range(1, self._nx):

                        # Вычисляем новое значение
                        num = (u_prev[j, i+1] + u_prev[j, i-1]) / hx2 + (u_prev[j+1, i] + u_prev[j-1, i]) / hy2
                        u[j, i] = num / denom_mpi

                self._post_solution(u, cur_iter)

                # Вычисление погрешности и критерия остановки
                end_crit = self._pogr_step(u_prev, u)

                # Обновление u_prev с помощью метода .copy()
                u_prev = u.copy() 

                cur_iter += 1

            print(f"Закончили вычисления на итерации {cur_iter-1} с значением критерия остановки {end_crit}")

        else: # Зейдель (Gauss-Seidel) или Верхняя релаксация (SOR)
            
            param = 1.0 # Зейдель

            if scheme_type == 3:
                param = w # Верхняя релаксация

            while (end_crit > eps):
                
                # Итерация Зейделя/SOR: неявное вычисление, использует новые значения u[j, i-1] и u[j-1, i]
                for j in range(1, self._ny):
                    for i in range(1, self._nx):
                        
                        # Вычисление "Зейделевского" значения u_star
                        num_gs = (u_prev[j, i+1] + u[j, i-1]) / hx2 + (u_prev[j+1, i] + u[j-1, i]) / hy2
                        u_star = num_gs / denom_gs_sor

                        # Применение формулы SOR (при param=1 это Зейдель)
                        u[j, i] = (1.0 - param) * u_prev[j, i] + param * u_star

                self._post_solution(u, cur_iter)

                # Вычисление погрешности и критерия остановки
                end_crit = self._pogr_step(u_prev, u)

                # Обновление u_prev с помощью метода .copy()
                u_prev = u.copy()

                cur_iter += 1

            print(f"Закончили вычисления на итерации {cur_iter-1} с значением критерия остановки {end_crit}")

if __name__ == "__main__":
    # Папки для сохранения результатов для каждой схемы
    # Создаем отдельные папки для сравнения результатов разных схем
    path_mpi = os.path.join(DATA_PATH, 'mpi_np')
    path_zeidel = os.path.join(DATA_PATH, 'zeidel_np')
    path_relax = os.path.join(DATA_PATH, 'relax_np')
    
    os.makedirs(path_mpi, exist_ok=True)
    os.makedirs(path_zeidel, exist_ok=True)
    os.makedirs(path_relax, exist_ok=True)

    x_steps_val = 20
    y_steps_val = 20
    relaxation_param = 1.5

    solver_mpi = ELLIP_SOLVER(saving_path=path_mpi, x_steps=x_steps_val, y_steps=y_steps_val)
    solver_zeidel = ELLIP_SOLVER(saving_path=path_zeidel, x_steps=x_steps_val, y_steps=y_steps_val)
    solver_relax = ELLIP_SOLVER(saving_path=path_relax, x_steps=x_steps_val, y_steps=y_steps_val)

    print("--- Решение методом МПИ (Jacobi) ---")
    solver_mpi.solve(scheme_type=1) 

    print("\n--- Решение методом Зейделя (Gauss-Seidel) ---")
    solver_zeidel.solve(scheme_type=2)

    print(f"\n--- Решение методом Верхней Релаксации (SOR, w={relaxation_param}) ---")
    solver_relax.solve(scheme_type=3, w=relaxation_param)
    
    # Визуализация и сравнение
    visual_paths = {
        'МПИ': path_mpi,
        'Зейдель': path_zeidel,
        'SOR': path_relax
    }   

    for name, path in visual_paths.items():
        print(f"\nВизуализация для: {name}")
        visualise(path=path, title=name)
import os
import math

from copy import deepcopy
from visual import visualise
from tridiag import TRIDIAG_SOLVER

data_folder = "results"
DATA_PATH = os.path.join(os.path.split(os.path.realpath(__file__))[0], data_folder)
os.makedirs(DATA_PATH, exist_ok=True)

# Вариант 7
class DIM_SOLVER:
    def __init__(self, saving_path: str, x_steps: int = 10, y_steps: int = 10, max_t: float = 2.0, t_step: float = 0.1):
        """
        Папка сохранения результатов saving_path (не должно быть других .txt).
        
        Args:
            x_steps (int): Число шагов по x-координате.
            y_steps (int): Число шагов по y-координате.
            max_t (float): "Потолок" по времени.
            t_step (float): Размер шага по времени.
        """
        
        self._path = saving_path
        
        self._nx = x_steps
        self._xd = 1.0 / self._nx

        self._ny = y_steps
        self._yd = 1.0 / self._ny

        self._td = t_step
        self._t_steps = int(max_t // self._td)

        self._edge_left = lambda y,t: 0*y
        self._edge_right = lambda y,t: y*math.cos(t)
        self._edge_bottom = lambda x,t: 0*x
        self._edge_top = lambda x,t: x*math.cos(t)

        self._start_cond = lambda x,y: x*y

        self._true_sol = lambda x,y,t: x*y*math.cos(t)

    def _cleanup_dir(self):
        """
        Очистить папку результатов от "*.txt".
        """
        txt_files = [f for f in os.listdir(self._path) if f.endswith('.txt')]
    
        for file in txt_files:
            file_path = os.path.join(self._path, file)
            os.remove(file_path)
            # print(f"Removed: {file_path}")

    def _write_res(self, u:list[list[float]], ts: list[list[float]], cur_t:float, iter:int, pogr: float):
        """
        Записать время, вычисленное и точное значение для каждой точки в файл.
        """
        
        ind_path = self._path + '/' + str(iter) + ".txt"
        with open(ind_path, "w") as f:
            f.write(str(cur_t) + '\n')
            for j in range (self._ny+1):
                for i in range (self._nx+1):
                    f.write(str(j*self._yd) + ' ' + str(i*self._xd) + ' ' +  str(u[j][i]) + ' ' + str(ts[j][i]) + '\n')

        pogr_path = self._path + "/p.txt"
        with open(pogr_path, "a") as f:
            f.write(str(cur_t) + ' ' + str(pogr) + '\n')

    def _pogr_step(self, u: list[float], ts: list[float]):
        """
        Рассчитать погрешность для данного времени.
        """
        pogr = 0
        for j in range (self._ny+1):
            for i in range (self._nx+1):
                pogr = max(pogr, abs(u[j][i]-ts[j][i]))
        return pogr
        
    def _post_solution(self, u:list[float], t:float, iter:float):
        """
        Сделать действия после шага решения.
        """
        cur_true = []
        for j in range(self._ny+1):
            cur_true.append([self._true_sol(self._xd*i, self._yd*j, t) for i in range(self._nx+1)])
                
        pogr = self._pogr_step(u, cur_true)
        self._write_res(u, cur_true, t, iter, pogr)

    def solve(self, scheme_type: int = 1):
        """
        Вычислить и сохранить решение.

        Args:
            scheme_type (str): Тип метода. 1 - переменных направлений, 2 - дробных шагов.
        """

        if scheme_type not in [1,2]:
            raise ValueError("Неверно указаны параметры решателя!")
        
        self._cleanup_dir()

        u = []

        for j in range(self._ny+1):
            u.append([self._start_cond(self._xd*i, self._yd*j) for i in range(self._nx+1)])

        u_prev = deepcopy(u)
        
        if scheme_type == 1: # МПН

            for k in range (0, self._t_steps+1):

                cur_t = (k+0.5)*self._td

                # Левая и правая границы
                for j in range(self._ny+1):
                    u[j][0] = self._edge_left(self._yd*j, cur_t)
                    u[j][self._nx] = self._edge_right(self._yd*j, cur_t)

                # Верхняя и нижняя границы
                for i in range(1, self._nx):
                    u[0][i] = self._edge_bottom(self._xd*i, cur_t)
                    u[self._ny][i] = self._edge_top(self._xd*i, cur_t)

                # Выбираем направление для неявной схемы
                for j in range(1, self._ny):
                    a = [0]*(self._nx-1); b = [0]*(self._nx-1); c = [0]*(self._nx-1); d = [0]*(self._nx-1)
                        
                    for i in range(self._nx-1):

                        a[i] = -1/self._xd**2
                        b[i] = 2/self._xd**2 + 2/self._td
                        c[i] = -1/self._xd**2
                        d[i] = 2/self._td*u_prev[j][i] + (1/self._yd**2)*(u_prev[j+1][i]-2*u_prev[j][i]+u_prev[j-1][i]) - (self._xd*i)*(self._yd*j)*math.sin(cur_t)

                    d[0] -= a[0]*u[j][0]
                    a[0] = 0

                    d[self._nx-2] -= c[self._nx-2]*u[j][self._nx]
                    c[self._nx-2] = 0

                    progon_solver = TRIDIAG_SOLVER(a,b,c,d)

                    progon_solution = progon_solver.solve()

                    for i in range(1, self._nx):
                        u[j][i] = progon_solution[i-1]

                u_prev = deepcopy(u)

                cur_t += 0.5*self._td

                # Левая и правая границы
                for j in range(self._ny+1):
                    u[j][0] = self._edge_left(self._yd*j, cur_t)
                    u[j][self._nx] = self._edge_right(self._yd*j, cur_t)

                # Верхняя и нижняя границы
                for i in range(1, self._nx):
                    u[0][i] = self._edge_bottom(self._xd*i, cur_t)
                    u[self._ny][i] = self._edge_top(self._xd*i, cur_t)

                for i in range(1, self._nx):
                    a = [0]*(self._ny-1); b = [0]*(self._ny-1); c = [0]*(self._ny-1); d = [0]*(self._ny-1)
                        
                    for j in range(self._ny-1):

                        a[j] = -1/self._yd**2
                        b[j] = 2/self._yd**2 + 2/self._td
                        c[j] = -1/self._yd**2
                        d[j] = 2/self._td*u_prev[j][i] + (1/self._xd**2)*(u_prev[j][i+1]-2*u_prev[j][i]+u_prev[j][i-1]) - (self._xd*i)*(self._yd*j)*math.sin(cur_t - 0.5*self._td)

                    d[0] -= a[0]*u[0][i]
                    a[0] = 0

                    d[self._ny-2] -= c[self._ny-2]*u[self._ny][i]
                    c[self._ny-2] = 0

                    progon_solver = TRIDIAG_SOLVER(a,b,c,d)

                    progon_solution = progon_solver.solve()

                    for j in range(1, self._ny):
                        u[j][i] = progon_solution[j-1]
                
                u_prev = deepcopy(u)

                self._post_solution(u, cur_t, k)

        if scheme_type == 2: # МДШ

            for k in range (0, self._t_steps+1):

                cur_t = (k+0.5)*self._td

                # Левая и правая границы
                for j in range(self._ny+1):
                    u[j][0] = self._edge_left(self._yd*j, cur_t)
                    u[j][self._nx] = self._edge_right(self._yd*j, cur_t)

                # Верхняя и нижняя границы
                for i in range(1, self._nx):
                    u[0][i] = self._edge_bottom(self._xd*i, cur_t)
                    u[self._ny][i] = self._edge_top(self._xd*i, cur_t)

                # Выбираем направление для неявной схемы
                for j in range(1, self._ny):
                    a = [0]*(self._nx-1); b = [0]*(self._nx-1); c = [0]*(self._nx-1); d = [0]*(self._nx-1)
                        
                    for i in range(self._nx-1):

                        a[i] = -1/self._xd**2
                        b[i] = 2/self._xd**2 + 1/self._td
                        c[i] = -1/self._xd**2
                        d[i] = 1/self._td*u_prev[j][i] - (self._xd*i)*(self._yd*j)*math.sin(k*self._td)/2

                    d[0] -= a[0]*u[j][0]
                    a[0] = 0

                    d[self._nx-2] -= c[self._nx-2]*u[j][self._nx]
                    c[self._nx-2] = 0

                    progon_solver = TRIDIAG_SOLVER(a,b,c,d)

                    progon_solution = progon_solver.solve()

                    for i in range(1, self._nx):
                        u[j][i] = progon_solution[i-1]

                u_prev = deepcopy(u)

                cur_t += 0.5*self._td

                # Левая и правая границы
                for j in range(self._ny+1):
                    u[j][0] = self._edge_left(self._yd*j, cur_t)
                    u[j][self._nx] = self._edge_right(self._yd*j, cur_t)

                # Верхняя и нижняя границы
                for i in range(1, self._nx):
                    u[0][i] = self._edge_bottom(self._xd*i, cur_t)
                    u[self._ny][i] = self._edge_top(self._xd*i, cur_t)

                for i in range(1, self._nx):
                    a = [0]*(self._ny-1); b = [0]*(self._ny-1); c = [0]*(self._ny-1); d = [0]*(self._ny-1)
                        
                    for j in range(self._ny-1):

                        a[j] = -1/self._yd**2
                        b[j] = 2/self._yd**2 + 1/self._td
                        c[j] = -1/self._yd**2
                        d[j] = 1/self._td*u_prev[j][i] - (self._xd*i)*(self._yd*j)*math.sin(cur_t)/2

                    d[0] -= a[0]*u[0][i]
                    a[0] = 0

                    d[self._ny-2] -= c[self._ny-2]*u[self._ny][i]
                    c[self._ny-2] = 0

                    progon_solver = TRIDIAG_SOLVER(a,b,c,d)

                    progon_solution = progon_solver.solve()

                    for j in range(1, self._ny):
                        u[j][i] = progon_solution[j-1]
                
                u_prev = deepcopy(u)

                self._post_solution(u, cur_t, k)

if __name__ == "__main__":
    solver = DIM_SOLVER(saving_path=DATA_PATH, x_steps = 10, y_steps = 20, max_t = 10.0)

    for i in range (1,3):
        solver.solve(i)
        visualise(path=DATA_PATH)
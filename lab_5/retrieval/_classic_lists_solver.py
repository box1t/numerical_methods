# import os
# import math

# from copy import deepcopy
# from tridiag import TRIDIAG_SOLVER
# from visual import visualise

# data_folder = "results"
# DATA_PATH = os.path.join(os.path.split(os.path.realpath(__file__))[0], data_folder)
# os.makedirs(DATA_PATH, exist_ok=True)

# # Вариант 7
# class PARAB_SOLVER:
#     def __init__(self, saving_path, x_steps = 10, max_t = 2.0):
#         """
#         Папка сохранения результатов saving_path (не должно быть других .txt).

#         Разбиение по x-координате x_steps.

#         Конечное время max_t.
#         """

#         if not x_steps >= 3 and max_t > 0:
#             raise ValueError("Неверно указаны шаги!")

#         self._path = saving_path

#         self._n = x_steps
#         self._xd = math.pi / self._n
#         self._td = 0.5*self._xd**2
#         self._t_steps = int(max_t // self._td)

#         self._start_cond = lambda x: math.sin(x)
#         self._true_sol = lambda x, t: math.exp(-0.5*t)*math.sin(x)

#     def _write_res(self, u:list[float], ts: list[float], t:float, num: int, pogr: float):
#         """
#         Записать время, вычисленное и точное значение для каждой точки в файл.
#         """

#         ind_path = self._path + '/' + str(num) + ".txt"
#         with open(ind_path, "w") as f:
#             f.write(str(t) + '\n')
#             for i in range (self._n+1):
#                 cur_x = i*self._xd
#                 f.write(str(cur_x) + ' ' + str(u[i]) + ' ' + str(ts[i]) + '\n')

#         pogr_path = self._path + "/p.txt"
#         with open(pogr_path, "a") as f:
#             f.write(str(t) + ' ' + str(pogr) + '\n')

#     def _cleanup_dir(self):
#         """
#         Очистить папку результатов от "*.txt".
#         """
#         txt_files = [f for f in os.listdir(self._path) if f.endswith('.txt')]
    
#         for file in txt_files:
#             file_path = os.path.join(self._path, file)
#             os.remove(file_path)
#             # print(f"Removed: {file_path}")

#     def _pogr_step(self, u: list[float], ts: list[float]):
#         """
#         Рассчитать погрешность для данного времени.
#         """
#         pogr = 0
#         for i in range (self._n+1):
#             pogr = max(pogr, abs(u[i]-ts[i]))
#         return pogr

#     def _post_solution(self, u:list[float], t:float, num: int):
#         """
#         Сделать действия после шага решения.
#         """
#         cur_true = [self._true_sol(i*self._xd, t) for i in range (self._n+1)]
#         pogr = self._pogr_step(u, cur_true)
#         self._write_res(u, cur_true,  t, num, pogr)

# # где следует добавить проверку условия устойчивости?
# # как измеряется "n-точечность" аппроксимации? почему здесь везде 2-точечная?
#     def solve(self, scheme_type: int = 1, approx_type: int = 1, theta = 0.5):
#         """
#         Вычислить и сохранить решение.

#         Тип схемы scheme_type: 1 - явная, 2 - неявная, 3 - Кранка-Николсона.

#         Тип аппроксимации approx_type: 1 - 2точ1пор, 2 - 3точ2пор, 3 - 2точ2пор.

#         Вес в схеме Кранка-Николсона theta: [0;1]
#         """

#         if scheme_type not in [1,2,3] and approx_type not in [1,2,3] and theta >= 0 and theta <= 1:
#             raise ValueError("Неверно указаны параметры решателя!")

#         self._cleanup_dir()
#         # почему сначала ищется u_cur, учитывая тип аппроксимации? верно ли понимаю, что это формулы 5.38 и 5.39, из которых достаем u_j_k?
#         # если это так, тогда остаточный член вычитается как часть нач условия?
#         # почему в явной схеме вычитаемое меньше, чем в неявной? в чем разница между tau^2 и tau? верно ли понимаю, что взятие второй производной по tau сводит слагаемое к нулю?
#         #  
#         # почему для аппроксимации вычитаем из значения функции в t=0 значение в t=1?
#         # почему изначально делаем u_cur = u_prev?
#         # при первом типе аппроксимации / втором цикл распространяется только для начального условия? или как это работает?

#         u = [self._start_cond(i*self._xd) for i in range(0, self._n+1)]

#         u_prev = deepcopy(u)

#         if scheme_type == 1:

#             for j in range (1, self._t_steps+1):
#                 cur_t = j * self._td
#                 for i in range(1, self._n):
#                     cur_x = self._xd*i

#                     u[i] = u_prev[i] + self._td*(u_prev[i+1]-2*u_prev[i]+u_prev[i-1])/(self._xd**2) \
#                         + 0.5*self._td*math.exp(-0.5*(cur_t-self._td))*math.sin(cur_x)

#                 # использует ближайшую (к началу) точку сетки для аппроксимации производной на границах сетки. 
#                 # отсюда находятся значения сеточной функции на границах. разность берется по формуле конечной разности: текущее знач. - предыдущее знач.
                
#                 if approx_type == 1:
#                     u[0] = u[1] - self._xd*math.exp(-0.5*cur_t)
#                     u[self._n] = u[self._n-1] - self._xd*math.exp(-0.5*cur_t)

                
#                 # используется центральная разность.
#                 # 3-ья точка - дополнительная фиктивная вне расчетной области. 
#                 # где логично применять трехточечную? удобно ли это? 


#                 # 3точ2пор
#                 elif approx_type == 2:
#                     u[0] = 1/3 * (4 * u[1] - u[2] - 2 * self._xd * math.exp(-0.5 * cur_t))
#                     u[self._n] = 1/3 * (-u[self._n-2] + 4 * u[self._n-1] - 2 * self._xd * math.exp(-0.5 * cur_t))

#                 # самый сложный вариант. 2точ2пор, но учитывает значение функции в нач. точке и еще в двух соседних точках.

#                 elif approx_type == 3:
#                     u[0] = u[1] - self._xd*math.exp(-0.5 * cur_t) + self._xd**2 / 2 * u_prev[0]/self._td
#                     u[0] /= 1 + (self._xd ** 2) / (2 * self._td)

#                     u[self._n] = u[self._n-1] - self._xd*math.exp(-0.5 * cur_t) \
#                         + self._xd**2 / 2 * (u_prev[self._n] / self._td)
#                     u[self._n] /= 1 + (self._xd**2) / (2 * self._td)

#                 u_prev = deepcopy(u)

#                 self._post_solution(u, cur_t, j)

#         else:
#             theta = 1
#             if scheme_type == 3:
#                 theta = 0.5

#             for j in range (1, self._t_steps+1):
#                 cur_t = j*self._td
#                 a = [0]*(self._n+1); b = [0]*(self._n+1); c = [0]*(self._n+1); d = [0]*(self._n+1)

#                 for i in range(1, self._n):
#                     cur_x = self._xd*i

#                     a[i] = -self._td*theta
#                     b[i] = self._xd ** 2 + 2 * self._td*theta
#                     c[i] = -self._td * theta
#                     d[i] = u[i]*(self._xd ** 2) + self._td * (1 - theta) * (u[i+1] - 2 * u[i] + u[i-1]) \
#                         + 0.5 * (self._xd ** 2) * self._td * math.sin(cur_x) * (theta * math.exp(-0.5 * cur_t)+(1 - theta) * math.exp(-0.5 * (cur_t - self._td)))

#                 if approx_type == 1:
#                     b[0] = -1
#                     c[0] = 1
#                     d[0] = self._xd*math.exp(-0.5 * cur_t)

#                     a[self._n] = -1
#                     b[self._n] = 1
#                     d[self._n] = -self._xd*math.exp(-0.5 * cur_t)

#                 elif approx_type == 2:
#                     b[0] = -3 - a[1] / self._td / theta
#                     c[0] = 4 - b[1] / self._td / theta
#                     d[0] = 2 * self._xd*math.exp(-0.5 * cur_t) - d[1] / self._td / theta

#                     a[self._n] = -4 + b[self._n-1] / self._td / theta
#                     b[self._n] = 3 + c[self._n-1] / self._td / theta
#                     d[self._n] = -2*self._xd*math.exp(-0.5*cur_t) + d[self._n-1] / self._td / theta

#                 elif approx_type == 3:
#                     b[0] = 1 + (self._xd ** 2) / (2 * self._td)
#                     c[0] = -1
#                     d[0] = -self._xd*math.exp(-0.5 * cur_t)

#                     a[self._n] = -1
#                     b[self._n] = 1 + (self._xd ** 2) / (2 * self._td)
#                     d[self._n] = - self._xd * math.exp(-0.5 * cur_t) \
#                         + self._xd ** 2 / 2 * (u_prev[self._n] / self._td)

#                 progon = TRIDIAG_SOLVER(a,b,c,d)

#                 u = progon.solve()

#                 u_prev = deepcopy(u)

#                 self._post_solution(u, cur_t, j)

# if __name__ == "__main__":
#     solver = PARAB_SOLVER(saving_path=DATA_PATH)

#     solver.solve(3, 1)
#     visualise(path=DATA_PATH)

#     for i in range (1,4):
#         for j in range (1,4):
#             solver.solve(i,j)
#             visualise(path=DATA_PATH)



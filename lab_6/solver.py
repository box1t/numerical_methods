import os
import math

from copy import deepcopy
from tridiag import TRIDIAG_SOLVER
from visual import visualise

data_folder = "results"
DATA_PATH = os.path.join(os.path.split(os.path.realpath(__file__))[0], data_folder)
os.makedirs(DATA_PATH, exist_ok=True)

# Вариант 7
class HYPERB_SOLVER:
    def __init__(self, saving_path, x_steps = 10, max_t = 2.0):
        """
        Папка сохранения результатов saving_path (не должно быть других .txt).
        
        Разбиение по x-координате x_steps.

        Конечное время max_t.
        """

        if not x_steps >= 3 and max_t > 0:
            raise ValueError("Неверно указаны шаги!")
        
        self._path = saving_path
        
        self._n = x_steps
        self._xd = math.pi / 2 / self._n
        self._td = 0.5*self._xd
        self._t_steps = int(max_t // self._td)

        self._start_cond = lambda x: math.exp(-x)*math.cos(x)
        self._true_sol = lambda x, t: math.exp(-t-x)*math.cos(x)*math.cos(2*t)

    def _write_res(self, u:list[float], ts: list[float], t:float, num: int, pogr: float):
        """
        Записать время, вычисленное и точное значение для каждой точки в файл.
        """
        
        ind_path = self._path + '/' + str(num) + ".txt"
        with open(ind_path, "w") as f:
            f.write(str(t) + '\n')
            for i in range (self._n+1):
                cur_x = i*self._xd
                f.write(str(cur_x) + ' ' + str(u[i]) + ' ' + str(ts[i]) + '\n')

        pogr_path = self._path + "/p.txt"
        with open(pogr_path, "a") as f:
            f.write(str(t) + ' ' + str(pogr) + '\n')

    def _cleanup_dir(self):
        """
        Очистить папку результатов от "*.txt".
        """
        txt_files = [f for f in os.listdir(self._path) if f.endswith('.txt')]
    
        for file in txt_files:
            file_path = os.path.join(self._path, file)
            os.remove(file_path)
            # print(f"Removed: {file_path}")

    def _pogr_step(self, u: list[float], ts: list[float]):
        """
        Рассчитать погрешность для данного времени.
        """
        pogr = 0
        for i in range (self._n+1):
            pogr = max(pogr, abs(u[i]-ts[i]))
        return pogr
        
    def _post_solution(self, u:list[float], t:float, num: int):
        """
        Сделать действия после шага решения.
        """
        cur_true = [self._true_sol(i*self._xd, t) for i in range (self._n+1)]
        pogr = self._pogr_step(u, cur_true)
        self._write_res(u, cur_true,  t, num, pogr) 


# где следует добавить проверку условия устойчивости?
# как измеряется "n-точечность" аппроксимации? почему здесь везде 2-точечная?
    def solve(self, scheme_type: int = 1, approx_type: int = 1):
        """
        Вычислить и сохранить решение.

        Схема scheme_type: 1 - явная, 2 - неявная.
        
        Аппроксимация approx_type: 1 - 2точ1пор, 2 - 2точ2пор.
        """
        if scheme_type not in [1,2] and approx_type not in [1,2]:
            raise ValueError("Неверно указаны параметры решателя!")
        
        self._cleanup_dir()

        # Для t=0
        u_prev = [self._start_cond(i * self._xd) for i in range(0, self._n + 1)]
        u_cur = u_new = deepcopy(u_prev)

        # почему сначала ищется u_cur, учитывая тип аппроксимации? верно ли понимаю, что это формулы 5.38 и 5.39, из которых достаем u_j_k?
        # если это так, тогда остаточный член вычитается как часть нач условия?
        # почему в явной схеме вычитаемое меньше, чем в неявной? в чем разница между tau^2 и tau? верно ли понимаю, что взятие второй производной по tau сводит слагаемое к нулю?
        #  
        # почему для аппроксимации вычитаем из значения функции в t=0 значение в t=1?
        # почему изначально делаем u_cur = u_prev?
        # при первом типе аппроксимации / втором цикл распространяется только для начального условия? или как это работает?


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
        
        # явная схема -> отсутствие прогоночных коэф, вычисление без решения слау???
        # !!. должно ли быть ограничение "порядок аппроксимации равен двум," для явной схемы? (стр. 14, абзац 1). нет. это определение схемы аппрокс.
        # следует ли в таком случае установить первичность выбора типа схемы над типом аппрокс? и как это осуществить?
        # почему аппроксимация уравнения не противоречит аппрокс начальных условий? нач усл это всё ещё внутренние узлы?
        # или внутренние узлы это только диффур?
        # что означает двухточечность аппроксимации? значит для перехода на другой временной слой задействованы лишь 2 точки? 
        # или n-точечность характеризует работу с граничными условиями, но не с внутренними узлами?
        # указанные типы аппроксимации служат для внутренних узлов или для граничных тоже? какие служат только для граничных? см. лаб 5.
        
         

        if scheme_type == 1:

            for j in range(1, self._t_steps + 1):

                cur_t = j * self._td

                u_new[0] = math.exp(-cur_t) * math.cos(2 * cur_t)
                #u_new[self._n] = 0

                # 1. почему здесь нет инициализации прогоночных коэф перед вложенным циклом?
                # 2. верно ли понимаю, что во втором вложенном цикле указан преобразованный вид уравнения? почему сразу выгодно так делать?
                # почему невыгодно указывать дифференциальное уравнение где-либо в классе, а используется сразу преобразованный в список вид под конечную разность?
                # 3. за что отвечает внешний цикл, а за что вложенный?
                # 4. 
                # почему во вложенный внутренний цикл напрямую заложено дифф ур-е в явной схеме?
                # скрести в отчете вложенный цикл и формулу из теории.



                for i in range(1, self._n):

                    u_new[i] = 2 * u_cur[i] - u_prev[i] + self._td * u_prev[i] + (self._td/self._xd) ** 2 * (u_cur[i+1] - 2 * u_cur[i] + u_cur[i-1]) \
                        + self._td ** 2 / self._xd * (u_cur[i+1] - u_cur[i-1]) - 3 * self._td ** 2 * u_cur[i]
                    u_new[i] /= (1 + self._td)

                self._post_solution(u_new, cur_t, j)

                u_prev = deepcopy(u_cur)
                u_cur = deepcopy(u_new)

        else:
        # неявная схема -> возникают (в результате чего-то) прогоночные коэф, вычисление с решением слау

            # 1. за что отвечает этот внешний цикл?
            for j in range (1, self._t_steps + 1):
                cur_t = j * self._td

                u_new[0] = math.exp(-cur_t) * math.cos(2 * cur_t)
                #u_new[self._n] = 0
                # 1. почему здесь есть инициализация прогоночных коэф перед вложенным циклом?
                a = [0]*(self._n-1); b = [0]*(self._n-1); c = [0]*(self._n-1); d = [0]*(self._n-1)

                # 1. за что отвечает этот внутренний цикл?
                # 2. на основе каких преобразований получены эти прогоночные коэф? укажи страницу?
                # заложено ли сюда исходное дифф ур-е, по аналогии с вложенным внутренним циклом в явной схеме?
                # как получить коэф. прогонки на основе исходного ДУ?

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

                self._post_solution(u_new, cur_t, j)

                u_prev = deepcopy(u_cur)
                u_cur = deepcopy(u_new)

if __name__ == "__main__":
    max_t = 5.0
    num_plots = 5

    method_titles = ["Явная схема", "Неявная схема"]
    approx_titles = ["2т1п", "2т2п"]

    solver = HYPERB_SOLVER(saving_path=DATA_PATH, max_t = max_t)

    for i in range (1,3):
        for j in range (1,3):
            solver.solve(i, j)
            visualise(path=DATA_PATH, num_plots=num_plots, title = method_titles[i-1] + ' ' + approx_titles[j-1])
import os
from copy import deepcopy

class TRIDIAG_SOLVER:
    def __init__(self, a: list = [], b:list = [], c:list = [], d:list = []):
        self._a = deepcopy(a)
        self._b = deepcopy(b)
        self._c = deepcopy(c)
        self._d = deepcopy(d)
        self._n = len(d)

    # def read_from_file(self, name):
    #     """Ввести матрицу из файла"""
    #     PATH = os.path.split(os.path.realpath(__file__))[0] + "/" + name
    #     with open(PATH, "r") as f:
    #         lines = f.readlines()
    #         self._n = len(lines)

    #         line_stripped = lines[0].strip().split(' ')
    #         self._a.append(0)
    #         self._b.append(float(line_stripped[0]))
    #         self._c.append(float(line_stripped[1]))
    #         self._d.append(float(line_stripped[2]))

    #         for i in range(1,self._n-1):
    #             line_stripped = lines[i].strip().split(' ')
    #             self._a.append(float(line_stripped[0]))
    #             self._b.append(float(line_stripped[1]))
    #             self._c.append(float(line_stripped[2]))
    #             self._d.append(float(line_stripped[3]))

    #         line_stripped = lines[self._n-1].strip().split(' ')
    #         self._a.append(float(line_stripped[0]))
    #         self._b.append(float(line_stripped[1]))
    #         self._c.append(0)
    #         self._d.append(float(line_stripped[2]))

    def check_conditions(self):
        """Проверить корректность и устойчивость."""
        for i in range (self._n):
            if not (abs(self._b[i]) > 0 and abs(self._b[i]) >= abs(self._a[i]) + abs(self._c[i])):
                return False
        return True

    def check_solution(self, x):
        """Проверить решение на правильность."""
        calc_d = []
        calc_d.append(self._b[0]*x[0]+self._c[0]*x[1])
        for i in range(1,self._n-1):
            calc_d.append(self._a[i]*x[i-1]+self._b[i]*x[i]+self._c[i]*x[i+1])
        calc_d.append(self._a[self._n-1]*x[self._n-2]+self._b[self._n-1]*x[self._n-1])
        for i in range (self._n):
            diff = calc_d[i] - self._d[i]
            if (diff > 1e-6):
                return False
        return True 

    def solve(self):
        """Решить методом прогонки."""
        # if self.check_conditions() == False:
        #     #print("Не выполняется проверка!")
        #     raise(ValueError("Не выполняется диагональное преобладание!"))
        A = []
        B = []
        x = [0 for _ in range(self._n)]
        A.append(-self._c[0]/self._b[0])
        B.append(self._d[0]/self._b[0])
        for i in range (1,self._n):
            A.append(-self._c[i]/(self._b[i]+self._a[i]*A[i-1]))
            B.append((self._d[i]-self._a[i]*B[i-1])/(self._b[i]+self._a[i]*A[i-1]))
        x[self._n-1] = B[self._n-1]
        for i in range (self._n-2, -1, -1):
            x[i] = A[i]*x[i+1]+B[i]
        if (self.check_solution(x)): 
            return x
        else:
            #return []
            raise(ValueError("Ошибка решения!"))

    def print_matrix(self):
        """Напечатать матрицу."""
        for i in range (self._n):
            print(self._a[i], self._b[i], self._c[i], "=", self._d[i], "\n")

if __name__ == "__main__":
    solver = TRIDIAG_SOLVER()

#     # решение с вводом через файл

#     # solver.read_from_file("tridiag_try.txt") 
    
#     # result = solver.solve()
#     # for i in range (len(result)):
#     #     print(f"x_{i} = {round(result[i],3)}\n")

#     # решение с вводом из кода

    a = [0, -1, -9, -1, 9]
    b = [-6, 13, -15, -7, -18]
    c = [5, 6, -4, 1, 0]
    d = [51, 100, -12, 47, -90]


    solver = TRIDIAG_SOLVER(a,b,c,d)

    result = solver.solve()
    for i in range (len(result)):
        print(f"x_{i} = {round(result[i],3)}\n")
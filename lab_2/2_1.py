import math
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return 4 ** x - 5 * x - 2

def fDer(x):
    return math.log(4) * 4 ** x - 5

def fDer2(x):
    return math.log(4) ** 2 * 4 ** x

# Эквивалентное уравнение для метода простых итераций
def phi(x):
    if 5 * x + 2 <= 0:
        return float('nan') 
    return math.log(5 * x + 2, 4)

def phiDer(x):
    if 5 * x + 2 <= 0:
        return float('nan')
    if math.log(4) == 0 or (5 * x + 2) == 0:
         return float('nan')
    return 5 / (math.log(4) * (5 * x + 2))


def newton(x0, eps, max_iter=1000):
    xPrev = x0
    iter = 0
    print(f"Начало метода Ньютона с x0 = {x0}, eps = {eps}")
    while (iter < max_iter):
        iter += 1
        f_val = f(xPrev)
        f_prime = fDer(xPrev)

        if f_prime == 0 or math.isnan(f_prime) or math.isinf(f_prime):
            print(f"Ошибка: Метод Ньютона: Производная равна нулю или не определена/бесконечна в точке x = {xPrev}. Итерация {iter}.")
            return None, iter

        xCur = xPrev - f_val / f_prime
        if iter > 1 and abs(xCur - xPrev) > 1e10 * abs(xPrev - (xPrev_prev if iter > 1 else xPrev)): 
             print(f"Предупреждение: Метод Ньютона: Возможно расходимость. Текущее значение x = {xCur}. Итерация {iter}.")

        if abs(xCur - xPrev) < eps:
            break
        xPrev_prev = xPrev
        xPrev = xCur

    if iter == max_iter:
         print(f"Предупреждение: Метод Ньютона: Достигнут лимит итераций ({max_iter}).")
    return xCur, iter

def simpleIterations(x0, eps, max_iter=1000):
    xPrev = x0
    iter = 0
    print(f"Начало метода простых итераций с x0 = {x0}, eps = {eps}")

    if math.isnan(phi(x0)) or math.isinf(phi(x0)):
        print(f"Ошибка: Метод простых итераций: Начальная точка {x0} вне области определения phi(x). Невозможно начать итерации.")
        return None, 0

    while (iter < max_iter):
        iter += 1
        xCur = phi(xPrev)

        if math.isnan(xCur) or math.isinf(xCur):
            print(f"Ошибка: Метод простых итераций: Выход за пределы допустимой области или некорректное значение в точке x = {xPrev}. Итерация {iter}.")
            return None, iter

        # Критерий остановки: по малости разности между последовательными приближениями
        if abs(xCur - xPrev) < eps:
            break

        if iter > 1 and abs(xCur - xPrev) > 1e10 * abs(xPrev - (xPrev_prev if iter > 1 else xPrev)):
             print(f"Предупреждение: Метод простых итераций: Возможно расходимость. Разность увеличивается. Итерация {iter}.")

        xPrev_prev = xPrev 
        xPrev = xCur

    if iter == max_iter:
         print(f"Предупреждение: Метод простых итераций: Достигнут лимит итераций ({max_iter}). Метод может не сойтись к заданной точности.")
    return xCur, iter


while True:
    try:
        a = float(input("Введите начало интервала [a]: "))
        b = float(input("Введите конец интервала [b]: "))
        if a >= b:
            print("Ошибка: Начало интервала должно быть меньше конца интервала.")
        else:
            if a <= -0.4:
                 print(f"Предупреждение: Начало интервала <= -0.4. Метод простых итераций с phi(x) = log4(5x+2) может не сойтись или дать ошибку, так как функция не определена для x <= -0.4.")

            break 
    except ValueError:
        print("Некорректный ввод. Пожалуйста, введите числовые значения.")

if f(a) * f(b) < 0:
    print(f"Условие f(a) * f(b) < 0 выполняется на интервале [{a}, {b}]. Корень существует.")
else:
    print(f"Предупреждение: Значения функции на концах интервала [{a}, {b}] имеют одинаковый знак или один из концов является корнем. В этом интервале может не быть корня или их может быть несколько четное число.")

der_a = fDer(a)
der_b = fDer(b)
if math.isnan(der_a) or math.isinf(der_a) or math.isnan(der_b) or math.isinf(der_b):
     print("Предупреждение: Производная не определена или бесконечна на одном или обоих концах интервала.")
elif np.sign(der_a) == np.sign(der_b):
     print(f"Производная сохраняет знак на концах интервала. Функция монотонна на [{a}, {b}].")
else:
     print(f"Предупреждение: Производная меняет знак на концах интервала. Функция не является монотонной на [{a}, {b}].")

if f(a) * f(b) < 0 and (math.isnan(der_a) or math.isinf(der_a) or math.isnan(der_b) or math.isinf(der_b) or np.sign(der_a) == np.sign(der_b)):
    print(f"На интервале [{a}, {b}] существует корень. Единственность: {'вероятна (если производная определена и сохраняет знак на всем интервале)' if not (math.isnan(der_a) or math.isinf(der_a) or math.isnan(der_b) or math.isinf(der_b)) and np.sign(der_a) == np.sign(der_b) else 'не гарантируется только по этим проверкам'}.")
else:
    print(f"На интервале [{a}, {b}] не гарантируется существование единственного корня на основе этих проверок.")

q_values = []
valid_range_for_q = False
if b > -0.4: 
     valid_range_for_q = True
     x_for_q_check = np.linspace(max(a, -0.4 + 1e-9), b, 200) 
     for x in x_for_q_check:
         phi_der_val = phiDer(x)
         if not math.isnan(phi_der_val) and not math.isinf(phi_der_val):
             q_values.append(abs(phi_der_val))


q = max(q_values) if q_values else float('inf')

if not valid_range_for_q:
     print(f"Информация о сходимости МПИ: Невозможно вычислить q на интервале [{a}, {b}], так как он вне области определения phi(x).")
elif q < 1:
    print(f"Информация о сходимости МПИ: Условие сходимости (|phi'(x)| <= q < 1) выполняется на части интервала [{a}, {b}] (там, где phi определена) с q = {q:.6f}. Сходимость гарантирована.")
else:
    print(f"Информация о сходимости МПИ: Условие сходимости (|phi'(x)| < 1) НЕ выполняется на части интервала [{a}, {b}] (где phi определена). q = {q:.6f}. Сходимость не гарантируется.")


# Ввод начальной точки для методов
while True:
    try:
        x0 = float(input(f"Введите начальное приближение x0 на интервале [{a}, {b}]: "))
        if a <= x0 <= b:
            # Дополнительная проверка: определена ли phi(x0) для метода простых итераций
            if not math.isnan(phi(x0)) and not math.isinf(phi(x0)):
                 break
            else:
                 print(f"Ошибка: Функция phi(x) не определена или бесконечна в начальной точке x0 = {x0}. Выберите другую точку.")
        else:
            print("Ошибка: Начальное приближение должно быть внутри заданного интервала.")
    except ValueError:
        print("Некорректный ввод. Пожалуйста, введите числовое значение.")

# Проверка условия f(x0) * f''(x0) > 0 для метода Ньютона
# Это условие для гарантированной монотонной сходимости к корню в интервале с пост. знаками f' и f'', не строго обязательно для самой работы метода
f_at_x0 = f(x0)
fDer2_at_x0 = fDer2(x0)
if not math.isnan(f_at_x0) and not math.isinf(f_at_x0) and not math.isnan(fDer2_at_x0) and not math.isinf(fDer2_at_x0):
    if f_at_x0 * fDer2_at_x0 > 0:
        print(f"Условие монотонной сходимости Ньютона f(x0) * f''(x0) > 0 выполняется для x0 = {x0}.")
    elif fDer(x0) == 0:
         print(f"Предупреждение: f'(x0) = 0 в начальной точке x0 = {x0}. Метод Ньютона может испытать трудности.")
    else:
        print(f"Предупреждение: Условие монотонной сходимости Ньютона f(x0) * f''(x0) > 0 НЕ выполняется для x0 = {x0}. Метод может сойтись к другому корню или разойтись.")
else:
    print(f"Предупреждение: Значения функции или второй производной в начальной точке x0 = {x0} не определены или бесконечны.")


eps = float(input("Введите требуемую точность (eps): "))

# --- Выполнение методов ---

print("\n--- Результаты ---")

# Метод Ньютона
newtonAns, iter_newton = newton(x0, eps)
if newtonAns is not None:
    print("Метод Ньютона")
    print("\tКорень: ", newtonAns)
    print("\tКоличество итераций: ", iter_newton)
else:
    print("Метод Ньютона не сошелся в пределах лимита итераций или возникла ошибка.")

# Метод простых итераций
# Выполняем МПИ, если начальная точка в области определения phi
if not math.isnan(phi(x0)) and not math.isinf(phi(x0)):
    simpleIterationsAns, iter_simple = simpleIterations(x0, eps) # Теперь МПИ не зависит от глобального q для остановки
    if simpleIterationsAns is not None:
        print("Метода простых итераций")
        print("\tКорень: ", simpleIterationsAns)
        print("\tКоличество итераций: ", iter_simple)
    else:
        print("Метод простых итераций не сошелся в пределах лимита итераций или возникла ошибка.")
else:
     print("Метод простых итераций не запускался, так как начальная точка вне области определения phi(x).")


print("\n--- Проверка найденных решений ---")

def verify_solution(x):
    return f(x)

if newtonAns is not None:
    verification_newton = verify_solution(newtonAns)
    print(f"Проверка корня методом Ньютона ({newtonAns:.6f}): f({newtonAns:.6f}) = {verification_newton:.2e}")
    if abs(verification_newton) < eps * 10: 
        print("Значение функции близко к нулю. Решение методом Ньютона подтверждено.")
    else:
        print("Значение функции не близко к нулю. Решение методом Ньютона может быть некорректным.")
else:
    print("Метод Ньютона не нашел решение для проверки.")

if simpleIterationsAns is not None:
    verification_simple = verify_solution(simpleIterationsAns)
    print(f"Проверка корня методом простых итераций ({simpleIterationsAns:.6f}): f({simpleIterationsAns:.6f}) = {verification_simple:.2e}")
    if abs(verification_simple) < eps * 10: 
        print("Значение функции близко к нулю. Решение методом простых итераций подтверждено.")
    else:
        print("Значение функции не близко к нулю. Решение методом простых итераций может быть некорректным.")
else:
     print("Метод простых итераций не нашел решение для проверки.")


# --- Построение графика ---

min_x = a - 0.5 
max_x = b + 0.5 
if newtonAns is not None:
    min_x = min(min_x, newtonAns - 0.1)
    max_x = max(max_x, newtonAns + 0.1)
if simpleIterationsAns is not None:
    min_x = min(min_x, simpleIterationsAns - 0.1)
    max_x = max(max_x, simpleIterationsAns + 0.1)

plot_min_x = min(min_x, -0.5) 
plot_max_x = max(max_x, 2.0)

plot_x_values = np.linspace(plot_min_x, plot_max_x, 400)
plot_y_values = f(plot_x_values)

plt.figure(figsize=(10, 6))
plt.plot(plot_x_values, plot_y_values, label='$4^x - 5x - 2 = 0$', color='blue')

# Отмечаем выбранный интервал на оси X
plt.axvspan(a, b, color='yellow', alpha=0.3, label=f'Интервал [{a}, {b}]')

area_of_def_end = max(plot_max_x, -0.4) 
if -0.4 < area_of_def_end:
    plt.axvspan(-0.4, area_of_def_end, color='lightgreen', alpha=0.1, label='Область определения $\phi(x)$ (x > -0.4)')


plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('График $4^x - 5x - 2 = 0$ и найденные корни')
plt.grid(True)
plt.axhline(0, color='red', linestyle='--', linewidth=0.7)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.7) 
plt.axvline(-0.4, color='purple', linestyle=':', linewidth=0.7, label='Граница области определения $\phi(x)$ (x = -0.4)') 


if newtonAns is not None:
    plt.plot(newtonAns, 0, 'go', markersize=8, label=f'Ньютон: x={newtonAns:.6f}')
if simpleIterationsAns is not None:
    plt.plot(simpleIterationsAns, 0, 'mo', markersize=8, label=f'Простые итерации: x={simpleIterationsAns:.6f}')
plt.plot(x0, f(x0), 'co', markersize=8, label=f'Начальная точка x0={x0}')


plt.legend()

plt.savefig('graph_2_1') 

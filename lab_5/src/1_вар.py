import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from lu_solver import lu_solve_modified

# ============== БЛОК 1: ФУНКЦИИ ДЛЯ ЧИСЛЕННОГО РЕШЕНИЯ ==================

def f(x, t):
    """Правая часть уравнения."""
    return 0.5 * np.exp(-0.5 * t) * np.cos(x)

def u_analytical(x, t):
    """Аналитическое решение."""
    return np.exp(-0.5 * t) * np.sin(x)

def solve_explicit(N, M, L, T_end):
    """
    Решение задачи явной схемой.
    """
    h = L / (N + 1)
    tau = T_end / M
    sigma = tau / h**2
    x_nodes = np.linspace(0, L, N + 2)
    u = np.zeros((M + 1, N + 2))
    u[0] = np.sin(x_nodes)

    if sigma > 0.5:
        print("Внимание: Явная схема может быть неустойчива (sigma > 0.5)!")

    for k in range(M):
        t_k = k * tau
        t_k_plus_1 = (k + 1) * tau
        
        u_next_inner = (u[k, 1:N+1] + sigma * (u[k, :N] - 2 * u[k, 1:N+1] + u[k, 2:N+2]) +
                        tau * f(x_nodes[1:N+1], t_k))
        u[k+1, 1:N+1] = u_next_inner
        
        # Граничные условия (1-го порядка)
        phi_0 = np.exp(-0.5 * t_k_plus_1)
        phi_L = -np.exp(-0.5 * t_k_plus_1)
        u[k+1, 0] = u[k+1, 1] - h * phi_0
        u[k+1, N+1] = u[k+1, N] + h * phi_L
    
    return u, x_nodes, u_analytical(x_nodes, T_end)

def solve_implicit(N, M, L, T_end):
    """
    Решение задачи неявной схемой с 2-точечной аппроксимацией ГУ 1-го порядка.
    """
    h = L / (N + 1)
    tau = T_end / M
    sigma = tau / h**2
    x_nodes = np.linspace(0, L, N + 2)
    u = np.zeros((M + 1, N + 2))
    u[0] = np.sin(x_nodes)
    
    # Система для внутренних узлов
    A = np.zeros((N, N))
    b = np.zeros(N)
    
    for j in range(N):
        A[j, j] = 1 + 2 * sigma
        if j > 0:
            A[j, j-1] = -sigma
        if j < N - 1:
            A[j, j+1] = -sigma

    for k in range(M):
        t_k = k * tau
        t_k_plus_1 = (k + 1) * tau
        
        b = u[k, 1:N+1] + tau * f(x_nodes[1:N+1], t_k_plus_1)
        
        # Граничные условия (1-го порядка)
        phi_0 = np.exp(-0.5 * t_k_plus_1)
        phi_L = -np.exp(-0.5 * t_k_plus_1)
        b[0] += sigma * (u[k, 0] + h * phi_0)
        b[N-1] += sigma * (u[k, N+1] - h * phi_L)
        
        # Решение системы
        try:
            u[k+1, 1:N+1] = lu_solve_modified(A, b)
        except ValueError as e:
            print(f"Ошибка на шаге {k+1}: {e}")
            return u, x_nodes, u_analytical(x_nodes, T_end)

        u[k+1, 0] = u[k+1, 1] - h * phi_0
        u[k+1, N+1] = u[k+1, N] + h * phi_L

    return u, x_nodes, u_analytical(x_nodes, T_end)

def solve_crank_nicolson(N, M, L, T_end):
    """
    Решение задачи схемой Кранка-Николсона с 2-точечной аппроксимацией ГУ 1-го порядка.
    """
    h = L / (N + 1)
    tau = T_end / M
    sigma = tau / h**2
    x_nodes = np.linspace(0, L, N + 2)
    u = np.zeros((M + 1, N + 2))
    u[0] = np.sin(x_nodes)

    A = np.zeros((N, N))
    for j in range(N):
        A[j, j] = 1 + sigma
        if j > 0:
            A[j, j-1] = -sigma / 2
        if j < N - 1:
            A[j, j+1] = -sigma / 2
            
    for k in range(M):
        t_half = (k + 0.5) * tau
        t_k_plus_1 = (k + 1) * tau
        
        b = (u[k, 1:N+1] + sigma / 2 * (u[k, :N] - 2 * u[k, 1:N+1] + u[k, 2:N+2]) +
             tau * f(x_nodes[1:N+1], t_half))
             
        # Граничные условия (1-го порядка)
        phi_0 = np.exp(-0.5 * t_k_plus_1)
        phi_L = -np.exp(-0.5 * t_k_plus_1)
        b[0] += sigma / 2 * (u[k, 0] + h * phi_0)
        b[N-1] += sigma / 2 * (u[k, N+1] - h * phi_L)
        
        try:
            u[k+1, 1:N+1] = lu_solve_modified(A, b)
        except ValueError as e:
            print(f"Ошибка на шаге {k+1}: {e}")
            return u, x_nodes, u_analytical(x_nodes, T_end)

        u[k+1, 0] = u[k+1, 1] - h * phi_0
        u[k+1, N+1] = u[k+1, N] + h * phi_L
    
    return u, x_nodes, u_analytical(x_nodes, T_end)

# ============== БЛОК 2: ФУНКЦИИ ДЛЯ ПОСТРОЕНИЯ ГРАФИКОВ ==================

def plot_2d_solution(x, u_num, u_an, T_end, title, output_dir):
    """
    Построение 2D-графика численного и аналитического решений.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(x, u_num, 'o-', label='Численное решение')
    plt.plot(x, u_an, '--', label='Аналитическое решение')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('u(x, t)')
    plt.legend()
    plt.grid(True)
    
    filename = f"{title.replace(' ', '_').replace('.', '')}_2D_solution.png"
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    graph_filepath = os.path.join(output_dir, filename)
    plt.savefig(graph_filepath, dpi=300)
    print(f"\nГрафик сохранен в файл {graph_filepath}")
    plt.close()

def plot_3d_solution(x, T, u, title, output_dir):
    """
    Построение 3D-графика эволюции решения.
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    X, T_grid = np.meshgrid(x, np.linspace(0, T, u.shape[0]))
    ax.plot_surface(X, T_grid, u, cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel('Пространственная переменная x')
    ax.set_ylabel('Временная переменная t')
    ax.set_zlabel('u(x, t)')
    
    filename = f"{title.replace(' ', '_').replace('.', '')}_3D_solution.png"
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    graph_filepath = os.path.join(output_dir, filename)
    plt.savefig(graph_filepath, dpi=300)
    print(f"\nГрафик сохранен в файл {graph_filepath}")
    plt.close()

# ============== ГЛАВНЫЙ БЛОК: ЗАПУСК ПРОГРАММЫ ==================

if __name__ == "__main__":
    
    # Параметры задачи
    L = np.pi
    T_end = 1.0
    N = 20  # Количество внутренних узлов
    M = 4000  # Количество временных шагов
    
    # Директория для сохранения
    OUTPUT_DIRECTORY = '/home/snowwy/snap/MAI/_math/8_Численные_методы/numerical_methods/lab_5/src/plots'

    # Явная схема
    u_explicit, x_exp, u_an_exp = solve_explicit(N, M, L, T_end)
    plot_2d_solution(x_exp, u_explicit[-1], u_an_exp, T_end, "Явная схема", OUTPUT_DIRECTORY)
    plot_3d_solution(x_exp, T_end, u_explicit, "Явная схема (эволюция)", OUTPUT_DIRECTORY)

    # Неявная схема
    u_implicit, x_imp, u_an_imp = solve_implicit(N, M, L, T_end)
    plot_2d_solution(x_imp, u_implicit[-1], u_an_imp, T_end, "Неявная схема", OUTPUT_DIRECTORY)
    plot_3d_solution(x_imp, T_end, u_implicit, "Неявная схема (эволюция)", OUTPUT_DIRECTORY)
    
    # Схема Кранка-Николсона
    u_cn, x_cn, u_an_cn = solve_crank_nicolson(N, M, L, T_end)
    plot_2d_solution(x_cn, u_cn[-1], u_an_cn, T_end, "Схема Кранка-Николсона", OUTPUT_DIRECTORY)
    plot_3d_solution(x_cn, T_end, u_cn, "Схема Кранка-Николсона (эволюция)", OUTPUT_DIRECTORY)
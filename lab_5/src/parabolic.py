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

def solve_explicit(N, M, L, T_end, bc_approx):
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
        
        phi_0 = np.exp(-0.5 * t_k_plus_1)
        phi_L = -np.exp(-0.5 * t_k_plus_1)

        if bc_approx == '1st_order_2_points':
            u[k+1, 0] = u[k+1, 1] - h * phi_0
            u[k+1, N+1] = u[k+1, N] + h * phi_L
        elif bc_approx == '2nd_order_3_points':
            u[k+1, 0] = (4 * u[k+1, 1] - u[k+1, 2] - 2 * h * phi_0) / 3
            u[k+1, N+1] = (4 * u[k+1, N] - u[k+1, N-1] + 2 * h * phi_L) / 3
    
    final_error = np.max(np.abs(u[-1] - u_analytical(x_nodes, T_end)))
    return u, x_nodes, final_error

def solve_implicit(N, M, L, T_end, bc_approx):
    """
    Решение задачи неявной схемой.
    """
    h = L / (N + 1)
    tau = T_end / M
    sigma = tau / h**2
    x_nodes = np.linspace(0, L, N + 2)
    u = np.zeros((M + 1, N + 2))
    u[0] = np.sin(x_nodes)
    
    for k in range(M):
        t_k_plus_1 = (k + 1) * tau
        phi_0 = np.exp(-0.5 * t_k_plus_1)
        phi_L = -np.exp(-0.5 * t_k_plus_1)
        
        if bc_approx == '1st_order_2_points':
            A = np.zeros((N, N))
            b = np.zeros(N)
            A[0, 0] = 1 + 2 * sigma
            A[0, 1] = -sigma
            b[0] = u[k, 1] + tau * f(x_nodes[1], t_k_plus_1) + sigma * (u[k, 0] + h * phi_0)
            for j in range(1, N - 1):
                A[j, j-1] = -sigma
                A[j, j] = 1 + 2 * sigma
                A[j, j+1] = -sigma
                b[j] = u[k, j+1] + tau * f(x_nodes[j+1], t_k_plus_1)
            A[N-1, N-2] = -sigma
            A[N-1, N-1] = 1 + 2 * sigma
            b[N-1] = u[k, N] + tau * f(x_nodes[N], t_k_plus_1) + sigma * (u[k, N+1] - h * phi_L)
            
            u_next_inner = lu_solve_modified(A, b)
            u[k+1, 1:N+1] = u_next_inner
            u[k+1, 0] = u[k+1, 1] - h * phi_0
            u[k+1, N+1] = u[k+1, N] + h * phi_L

        elif bc_approx == '2nd_order_3_points':
            A = np.zeros((N + 2, N + 2))
            b = np.zeros(N + 2)

            # Левая граница
            A[0, 0], A[0, 1], A[0, 2] = -3, 4, -1
            b[0] = 2 * h * phi_0
            
            # Внутренние узлы
            for j in range(1, N + 1):
                A[j, j-1] = -sigma
                A[j, j] = 1 + 2 * sigma
                A[j, j+1] = -sigma
                b[j] = u[k, j] + tau * f(x_nodes[j], t_k_plus_1)

            # Правая граница
            A[N+1, N-1], A[N+1, N], A[N+1, N+1] = 1, -4, 3
            b[N+1] = 2 * h * phi_L

            u[k+1] = lu_solve_modified(A, b)

        elif bc_approx == '2nd_order_2_points':
            A = np.zeros((N + 2, N + 2))
            b = np.zeros(N + 2)
            
            A[0, 0], A[0, 1] = 1 + 2 * sigma, -2 * sigma
            b[0] = u[k, 0] + tau * f(x_nodes[0], t_k_plus_1) + 2 * sigma * h * phi_0
            
            for j in range(1, N + 1):
                A[j, j-1] = -sigma
                A[j, j] = 1 + 2 * sigma
                A[j, j+1] = -sigma
                b[j] = u[k, j] + tau * f(x_nodes[j], t_k_plus_1)
                
            A[N+1, N], A[N+1, N+1] = -2 * sigma, 1 + 2 * sigma
            b[N+1] = u[k, N+1] + tau * f(x_nodes[N+1], t_k_plus_1) - 2 * sigma * h * phi_L
            
            u[k+1] = lu_solve_modified(A, b)
        
        if np.isnan(u[k+1]).any() or np.isinf(u[k+1]).any():
            raise ValueError("Решение расходится или содержит NaN/Inf.")

    final_error = np.max(np.abs(u[-1] - u_analytical(x_nodes, T_end)))
    return u, x_nodes, final_error

def solve_crank_nicolson(N, M, L, T_end, bc_approx):
    """
    Решение задачи схемой Кранка-Николсона.
    """
    h = L / (N + 1)
    tau = T_end / M
    sigma = tau / h**2
    x_nodes = np.linspace(0, L, N + 2)
    u = np.zeros((M + 1, N + 2))
    u[0] = np.sin(x_nodes)

    for k in range(M):
        t_half = (k + 0.5) * tau
        t_k_plus_1 = (k + 1) * tau
        phi_0 = np.exp(-0.5 * t_k_plus_1)
        phi_L = -np.exp(-0.5 * t_k_plus_1)

        if bc_approx == '1st_order_2_points':
            A = np.zeros((N, N))
            b = np.zeros(N)
            
            # Матрица
            A[0, 0] = 1 + sigma
            A[0, 1] = -sigma / 2
            for j in range(1, N - 1):
                A[j, j-1] = -sigma / 2
                A[j, j] = 1 + sigma
                A[j, j+1] = -sigma / 2
            A[N-1, N-2] = -sigma / 2
            A[N-1, N-1] = 1 + sigma
            
            # Вектор b
            b[0] = (u[k, 1] + sigma/2 * (u[k, 0] - 2*u[k, 1] + u[k, 2]) + 
                    tau * f(x_nodes[1], t_half) + sigma/2 * (u[k, 0] + h*phi_0))
            for j in range(1, N - 1):
                b[j] = (u[k, j+1] + sigma/2 * (u[k, j] - 2*u[k, j+1] + u[k, j+2]) + 
                        tau * f(x_nodes[j+1], t_half))
            b[N-1] = (u[k, N] + sigma/2 * (u[k, N-1] - 2*u[k, N] + u[k, N+1]) + 
                      tau * f(x_nodes[N], t_half) + sigma/2 * (u[k, N+1] - h*phi_L))
            
            u_next_inner = lu_solve_modified(A, b)
            u[k+1, 1:N+1] = u_next_inner
            u[k+1, 0] = u[k+1, 1] - h * phi_0
            u[k+1, N+1] = u[k+1, N] + h * phi_L

        elif bc_approx == '2nd_order_3_points':
            A = np.zeros((N + 2, N + 2))
            b = np.zeros(N + 2)
            
            A[0, 0], A[0, 1], A[0, 2] = -3, 4, -1
            b[0] = 2 * h * phi_0
            
            for j in range(1, N + 1):
                A[j, j-1] = -sigma / 2
                A[j, j] = 1 + sigma
                A[j, j+1] = -sigma / 2
                b[j] = (u[k, j] + sigma/2 * (u[k, j-1] - 2*u[k, j] + u[k, j+1]) + 
                        tau * f(x_nodes[j], t_half))

            A[N+1, N-1], A[N+1, N], A[N+1, N+1] = 1, -4, 3
            b[N+1] = 2 * h * phi_L
            
            u[k+1] = lu_solve_modified(A, b)

        elif bc_approx == '2nd_order_2_points':
            A = np.zeros((N + 2, N + 2))
            b = np.zeros(N + 2)
            
            A[0, 0], A[0, 1] = 1 + sigma, -sigma
            b[0] = (u[k, 0] + sigma/2 * (u[k, 0] - 2*u[k, 0] + u[k,1]) + 
                    tau*f(x_nodes[0], t_half) + sigma*h*phi_0)
            
            for j in range(1, N+1):
                A[j, j-1] = -sigma / 2
                A[j, j] = 1 + sigma
                A[j, j+1] = -sigma / 2
                b[j] = (u[k, j] + sigma/2 * (u[k, j-1] - 2*u[k, j] + u[k, j+1]) + 
                        tau * f(x_nodes[j], t_half))
                        
            A[N+1, N], A[N+1, N+1] = -sigma, 1 + sigma
            b[N+1] = (u[k, N+1] + sigma/2 * (u[k, N]-2*u[k,N+1]+u[k,N+1]) + 
                      tau*f(x_nodes[N+1], t_half) - sigma*h*phi_L)
            
            u[k+1] = lu_solve_modified(A, b)
            
        if np.isnan(u[k+1]).any() or np.isinf(u[k+1]).any():
            raise ValueError("Решение расходится или содержит NaN/Inf.")

    final_error = np.max(np.abs(u[-1] - u_analytical(x_nodes, T_end)))
    return u, x_nodes, final_error

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
    
    L = np.pi
    T_end = 1.0
    N = 20  
    M = 4000  
    
    OUTPUT_DIRECTORY = '/home/snowwy/snap/MAI/_math/8_Численные_методы/numerical_methods/lab_5/src/plots'
    
    schemes = {
        'Явная': solve_explicit,
        'Неявная': solve_implicit,
        'Кранка-Николсона': solve_crank_nicolson
    }
    
    bc_approximations = {
        '1st_order_2_points': 'Двухточечная аппроксимация 1-го порядка',
        '2nd_order_3_points': 'Трехточечная аппроксимация 2-го порядка',
        '2nd_order_2_points': 'Двухточечная аппроксимация 2-го порядка'
    }
    
    for scheme_name, solver_func in schemes.items():
        for bc_key, bc_name in bc_approximations.items():
            
            if scheme_name == 'Явная' and bc_key == '2nd_order_2_points':
                continue

            print(f"\n=======================================================")
            print(f"Расчет для: {scheme_name} схема с {bc_name}")
            print(f"=======================================================")
            
            try:
                u, x, final_error = solver_func(N, M, L, T_end, bc_key)
                print(f"Итоговая максимальная ошибка: {final_error:.6e}")
                
                title = f"{scheme_name} ({bc_name})"
                plot_2d_solution(x, u[-1], u_analytical(x, T_end), T_end, title, OUTPUT_DIRECTORY)
                plot_3d_solution(x, T_end, u, title, OUTPUT_DIRECTORY)
            except ValueError as e:
                print(f"Ошибка при расчете: {e}")
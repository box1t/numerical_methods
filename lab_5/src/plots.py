import os
import matplotlib.pyplot as plt
import numpy as np
from natsort import natsorted
from mpl_toolkits.mplot3d import Axes3D
import itertools

# --- Константы ---
DATA_FOLDER = "results"
SCHEME_STYLES = {'Явная': '-', 'Неявная': '--', 'Кранка-Николсона': ':'}
ERROR_STYLES = {'Явная': 'r-', 'Неявная': 'g--', 'Кранка-Николсона': 'b:'}

def load_solution_data(path: str) -> dict:
    """Загружает данные решения (x, u_solved, u_true) для всех времен t."""
    solutions = {} # {t: (x, u_solved, u_true)}

    data_files = [file for file in os.listdir(path) if file.endswith(".txt") and file[0] != 'p']
    data_files = natsorted(data_files)

    for file in data_files:
        full_path = os.path.join(path, file)
        with open(full_path, "r") as f:
            lines = f.readlines()
            try:
                t = float(lines[0].strip())
            except (ValueError, IndexError):
                continue 

            # Данные на остальных строках (x, u_solved, u_true)
            data_array = np.loadtxt(lines[1:], dtype=float)
            if data_array.ndim == 1:
                 data_array = data_array.reshape(1, -1)
            
            x, u_solved, u_true = data_array[:, 0], data_array[:, 1], data_array[:, 2]
            solutions[t] = (x, u_solved, u_true)
            
    return solutions

def load_error_data(path: str) -> tuple[list[float], list[float]]:
    """Загружает данные погрешности (t, error_vals)."""
    pogr_path = os.path.join(path, 'p.txt')
    t_pogr, pogr_vals = [], []

    if os.path.exists(pogr_path):
        data = np.loadtxt(pogr_path)
        if data.ndim > 1:
            t_pogr = data[:, 0]
            pogr_vals = data[:, 1]
        elif data.size > 0: # Для случая, когда записана только одна точка
            t_pogr = [data[0]]
            pogr_vals = [data[1]]
            
    return t_pogr, pogr_vals

def load_all_data(paths_dict: dict) -> tuple[dict, dict, list]:
    """Координирует загрузку данных для всех схем."""
    all_data = {}   # {name: solutions}
    all_pogr = {}   # {name: (t_pogr, pogr_vals)}
    
    # 1. Загрузка данных
    for name, path in paths_dict.items():
        all_data[name] = load_solution_data(path)
        all_pogr[name] = load_error_data(path)
    
    # 2. Получение общего списка времен (из первой схемы)
    if not all_data:
        return {}, {}, []
        
    first_key = next(iter(paths_dict))
    all_times = natsorted(all_data[first_key].keys())
    
    return all_data, all_pogr, all_times


# 📊 Модуль Визуализации (Отдельные, высокосвязанные функции)
# ---------------------------------------------------------------------

def plot_2d_comparison(all_data: dict, all_times: list, scheme_styles: dict, figure: plt.Figure):
    """
    Создает 2D график сравнения решений в выбранные моменты времени.
    """
    
    # Выбираем 3 момента времени для 2D сравнения
    n_times = len(all_times)
    chosen_indices = [n_times//4, n_times//2, n_times - 1]
    chosen_times = [all_times[i] for i in chosen_indices if i < n_times]
    
    time_colors = {chosen_times[i]: c for i, c in enumerate(['r', 'g', 'b'])}
    
    ax = figure.add_subplot(4, 1, (1, 3)) # Используем 3/4 вертикального пространства
    max_val_u = 0
    true_plotted = False
    
    # Находим ключ для Аналитического решения (первая схема в списке)
    first_key = next(iter(all_data))

    for t in chosen_times:
        current_color = time_colors[t]
        
        # --- Точное решение (u_true) ---
        if t in all_data[first_key]:
            x_true, _, u_true = all_data[first_key][t]
            max_val_u = max(max_val_u, np.max(u_true) if u_true.size > 0 else 0)

            label_true = f'Точное решение t={t:.3f}' if not true_plotted else None
            
            # Линия
            ax.plot(x_true, u_true, color='black', linestyle='-', linewidth=2.5, alpha=0.8, label=label_true, zorder=1) 
            # Маркеры
            ax.plot(x_true, u_true, color='black', marker='D', linestyle='None', markersize=5, 
                    markeredgecolor='black', markeredgewidth=1.5, zorder=3)
            true_plotted = True
            
        # --- Решения всех схем ---
        for name, data in all_data.items():
            if t in data:
                x_solved, u_solved, _ = data[t]
                style = scheme_styles.get(name)
                
                ax.plot(x_solved, u_solved, color=current_color, 
                        linestyle=style, linewidth=1.5, 
                        label=f'{name} t={t:.3f}')
                
                max_val_u = max(max_val_u, np.max(u_solved) if u_solved.size > 0 else 0)

    # Настройка графика
    ax.set_title("Сравнение схем в разные моменты времени T")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, T)")
    ax.set_ylim(-0.05, max_val_u * 1.1)
    ax.grid(True, linestyle='--')
    ax.legend(loc='upper right', ncol=3, fontsize='small')
    
    return max_val_u

def plot_3d_evolution(all_data: dict, all_times: list, max_val_u: float):
    """
    Создает отдельную фигуру с 3D графиками эволюции u(x,t).
    """
    # Схемы для отображения в 3D (Аналитическое, Явная, Неявная, Кранка-Николсона)
    scheme_names_for_3d = ['Аналитическое', 'Явная', 'Неявная', 'Кранка-Николсона']
    first_key = next(iter(all_data))
    
    # Создание словаря для удобного доступа к данным в цикле
    scheme_data_mapping = {
        'Аналитическое': all_data[first_key],
        'Явная': all_data.get('Явная'),
        'Неявная': all_data.get('Неявная'),
        'Кранка-Николсона': all_data.get('Кранка-Николсона')
    }
    
    figure_3d = plt.figure(figsize=(18, 10))

    for idx, scheme_name in enumerate(scheme_names_for_3d):
        current_scheme_data = scheme_data_mapping.get(scheme_name)

        if current_scheme_data:
            X_grid_list, T_grid_list, Z_grid_list = [], [], []
            
            for t_val in all_times:
                if t_val in current_scheme_data:
                    x_vals, u_solved_vals, u_true_vals = current_scheme_data[t_val]
                    
                    # Выбираем u_true для "Аналитического решения", u_solved для численных
                    z_vals = u_true_vals if scheme_name == 'Аналитическое' else u_solved_vals
                    
                    X_grid_list.append(x_vals)
                    T_grid_list.append(np.full_like(x_vals, t_val))
                    Z_grid_list.append(z_vals)
            
            if not Z_grid_list: # Пропускаем, если данных нет
                continue
                
            X_surf = np.array(X_grid_list)
            T_surf = np.array(T_grid_list)
            Z_surf = np.array(Z_grid_list)
            
            ax = figure_3d.add_subplot(2, 2, idx + 1, projection='3d')
            ax.view_init(elev=20, azim=-60)
            
            surf = ax.plot_surface(X_surf, T_surf, Z_surf, 
                                   cmap='plasma', 
                                   edgecolor='gray', 
                                   linewidth=0.2,
                                   rstride=2, cstride=2,
                                   antialiased=True
                                  )

            ax.set_title(f"{scheme_name} решение")
            ax.set_xlabel("x")
            ax.set_ylabel("t")
            ax.set_zlabel("u")
            ax.set_zlim(0, max_val_u * 1.1)
            
            figure_3d.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='u')
    
    figure_3d.tight_layout()


def plot_error_graph(all_pogr: dict, error_styles: dict, figure: plt.Figure):
    """
    Создает график погрешности в зависимости от времени.
    """
    ax = figure.add_subplot(4, 1, 4) # Занимает последнюю 1/4 часть
    
    max_pogr = 0
    for name, (t_pogr, pogr_vals) in all_pogr.items():
        if len(t_pogr) > 0:
            ax.plot(t_pogr, pogr_vals, error_styles.get(name, 'm-'), label=name)
            max_pogr = max(max_pogr, np.max(pogr_vals))

    ax.set_title("Погрешность (норма) в зависимости от времени для разных схем")
    ax.set_xlabel("t")
    ax.set_ylabel("||u_solved - u_true||_max")
    ax.set_ylim(0, max_pogr * 1.1)
    ax.grid(True, linestyle='--')
    ax.legend(loc='lower right', ncol=3)

# ---------------------------------------------------------------------
## 🚀 Главная Функция-Координатор (Низкая зависимость)
# ---------------------------------------------------------------------

def visualise_comparison(paths_dict: dict): 
    """
    Визуализирует сравнение схем, 3D эволюцию u(x,t) и график погрешности.
    """
    
    # 1. Загрузка всех данных
    all_data, all_pogr, all_times = load_all_data(paths_dict)
    
    if not all_data:
        print("Ошибка: Данные не найдены.")
        return
    
    # 2. Создание фигуры для 2D сравнения и погрешности
    figure_2d_pogr = plt.figure(figsize=(18, 20)) 
    
    # 3. 2D ГРАФИК СРАВНЕНИЯ
    # Возвращает максимальное значение, чтобы установить одинаковую Z-ось для 3D
    max_val_u = plot_2d_comparison(all_data, all_times, SCHEME_STYLES, figure_2d_pogr)
    
    # 4. 3D ЭВОЛЮЦИЯ РЕШЕНИЯ (в отдельной фигуре)
    plot_3d_evolution(all_data, all_times, max_val_u)
    
    # 5. ГРАФИК ПОГРЕШНОСТИ
    plot_error_graph(all_pogr, ERROR_STYLES, figure_2d_pogr)
    
    # Применяем tight_layout для 2D/Погрешности
    figure_2d_pogr.tight_layout() 
    
    # Отображаем обе фигуры
    plt.show(block=True)
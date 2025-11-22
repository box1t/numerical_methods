import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from natsort import natsorted
from mpl_toolkits.mplot3d import Axes3D
import itertools

DATA_FOLDER = "results"

# --- Вспомогательная функция для загрузки данных из одной папки ---
def _load_data_for_scheme(path):
    """Загружает данные решения и погрешности для одной схемы."""
    
    # 1. Загрузка файлов решения (u(x))
    data_files = [file for file in os.listdir(path) if file.endswith(".txt") and file[0] != 'p']
    data_files = natsorted(data_files)

    solutions = {} # {t: (x, u_solved, u_true)}

    for file in data_files:
        full_path = os.path.join(path, file)
        
        with open(full_path, "r") as f:
            lines = f.readlines()
            # Время на первой строке
            t_str = lines[0].strip()
            t = float(t_str)

            # Данные на остальных строках (x, u_solved, u_true)
            data_array = np.loadtxt(lines[1:], dtype=float)
            if data_array.ndim == 1: # Случай одного узла (если N=0), предотвращаем ошибку
                 data_array = data_array.reshape(1, -1)
            
            x = data_array[:, 0]
            u_solved = data_array[:, 1]
            u_true = data_array[:, 2]
            
            solutions[t] = (x, u_solved, u_true)
            
    # 2. Загрузка погрешности (p(t))
    pogr_path = os.path.join(path, 'p.txt')
    t_pogr = []
    pogr_vals = []
    if os.path.exists(pogr_path):
        data = np.loadtxt(pogr_path)
        if data.ndim > 1:
            t_pogr = data[:, 0]
            pogr_vals = data[:, 1]
        elif data.size > 0: # Для случая, когда записана только одна точка
            t_pogr = [data[0]]
            pogr_vals = [data[1]]

    return solutions, t_pogr, pogr_vals

def visualise_comparison(paths_dict: dict):
    """
    Визуализирует сравнение схем (2D графики), 3D эволюцию u(x,t) для 
    Аналитического и трех численных решений, и график погрешности.

    :param paths_dict: Словарь вида {'Название Схемы': 'Путь/к/результатам'}
    """
    
    # --- Загрузка всех данных ---
    all_data = {}
    all_pogr = {}
    
    first_key = next(iter(paths_dict))
    
    for name, path in paths_dict.items():
        solutions, t_pogr, pogr_vals = _load_data_for_scheme(path)
        all_data[name] = solutions
        all_pogr[name] = (t_pogr, pogr_vals)
    
    all_times = natsorted(all_data[first_key].keys())
    
    # Выбираем 3 момента времени для 2D сравнения
    n_times = len(all_times)
    chosen_indices = [n_times//4, n_times//2, n_times - 1]
    chosen_times = [all_times[i] for i in chosen_indices if i < n_times]
    
    
    # --- Создание общей фигуры ---
    # Макет: 4 ряда (1-й: 2D сравнение, 2-й и 3-й: 3D графики, 4-й: погрешность)
    figure = plt.figure(figsize=(18, 20)) 
    
    # ---------------------------------------------------------------------
    # 1. 2D ГРАФИК СРАВНЕНИЯ (РЯД 1)
    # ---------------------------------------------------------------------
    ax_comparison = figure.add_subplot(4, 1, (1, 3))

    scheme_styles = {'Явная': '-', 'Неявная': '--', 'Кранка-Николсона': ':'}
    time_colors = {chosen_times[0]: 'r', chosen_times[1]: 'g', chosen_times[2]: 'b'}
    
    true_plotted = False
    max_val_u = 0

    for t_index, t in enumerate(chosen_times):
        current_color = time_colors[t]
        
        # Точное решение (u_true)
        x_true, _, u_true = all_data[first_key][t]
        label_true = f'Точное решение t={t:.3f}' if not true_plotted else None
        
        ax_comparison.plot(x_true, u_true, 
                           color='black',            
                           linestyle='-',         
                           linewidth=2.5,            # Сделать контур толще
                           alpha=0.8,                # Небольшая прозрачность
                           label=label_true,
                           zorder=1)                 #

        ax_comparison.plot(x_true, u_true, color='black', marker='D', 
                        linestyle='None', markersize=5, 
                        markeredgecolor='black', 
                        markeredgewidth=1.5,
                        zorder=3, 
                        label=label_true)
                        
        true_plotted = True 
        
        # Решения всех схем
        for name, data in all_data.items():
            if t in data:
                x_solved, u_solved, u_true_at_t = data[t]
                
                style = scheme_styles.get(name)
                ax_comparison.plot(x_solved, u_solved, color=current_color, 
                                   linestyle=style, linewidth=1.5, label=f'{name} t={t:.3f}')
                
                # Обновление максимального значения
                max_val_u = max(max_val_u, np.max(u_solved), np.max(u_true_at_t))
    
    ax_comparison.set_title("Сравнение схем в разные моменты времени T")
    ax_comparison.set_xlabel("x")
    ax_comparison.set_ylabel("u(x, T)")
    ax_comparison.set_ylim(-0.05, max_val_u * 1.1)
    ax_comparison.grid(True, linestyle='--')
    ax_comparison.legend(loc='upper right', ncol=3, fontsize='small')

    

    # ---------------------------------------------------------------------
    # 2. 3D ЭВОЛЮЦИЯ РЕШЕНИЯ u(x,t) (РЯДЫ 2 и 3)
    # ---------------------------------------------------------------------
    
    # Схемы для отображения в 3D (порядок: Аналитическое, Явная, Неявная, Кранка-Николсона)
    scheme_names_for_3d = ['Аналитическое', 'Явная', 'Неявная', 'Кранка-Николсона']
    scheme_data_mapping = {
        'Аналитическое': all_data[first_key],
        'Явная': all_data.get('Явная'),
        'Неявная': all_data.get('Неявная'),
        'Кранка-Николсона': all_data.get('Кранка-Николсона')
    }
    
    # Запускаем новую фигуру для 3D графиков
    figure_3d = plt.figure(figsize=(18, 10))

    for idx, scheme_name in enumerate(scheme_names_for_3d):
        current_scheme_data = scheme_data_mapping[scheme_name]

        if current_scheme_data:
            X_grid_list = []
            T_grid_list = []
            Z_grid_list = [] 
            
            # Собираем данные X, T, Z для построения поверхности
            for t_val in all_times:
                if t_val in current_scheme_data:
                    x_vals, u_solved_vals, u_true_vals = current_scheme_data[t_val]
                    
                    X_grid_list.append(x_vals)
                    T_grid_list.append(np.full_like(x_vals, t_val))
                    
                    # Выбираем u_true для "Аналитического решения", u_solved для численных
                    if scheme_name == 'Аналитическое':
                        Z_grid_list.append(u_true_vals)
                    else:
                        Z_grid_list.append(u_solved_vals)
            
            # Конвертируем списки в numpy массивы
            X_surf = np.array(X_grid_list)
            T_surf = np.array(T_grid_list)
            Z_surf = np.array(Z_grid_list)
            
            # Подграфик 2x2 в отдельном окне
            ax = figure_3d.add_subplot(2, 2, idx + 1, projection='3d')
            
            # Устанавливаем угол обзора, чтобы соответствовал образцу
            ax.view_init(elev=20, azim=-60)
            
            # Построение поверхности
            surf = ax.plot_surface(X_surf, T_surf, Z_surf, 
                                   cmap='plasma', # Изменен на 'plasma' для лучшего контраста, 'viridis' или 'inferno' тоже подходят
                                   edgecolor='gray', 
                                   linewidth=0.2,
                                   rstride=2, cstride=2, # Более тонкая сетка
                                   antialiased=True
                                  )

            ax.set_title(f"{scheme_name} решение")
            ax.set_xlabel("x")
            ax.set_ylabel("t")
            ax.set_zlabel("u")
            ax.set_zlim(0, max_val_u * 1.1)
            
            # Добавляем Colorbar к каждому 3D-графику
            figure_3d.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='u')
    
    # Применяем tight_layout для 3D-графиков
    figure_3d.tight_layout()


    # ---------------------------------------------------------------------
    # 3. ГРАФИК ПОГРЕШНОСТИ (РЯД 4)
    # ---------------------------------------------------------------------
    
    # ax: позиция (4 ряда, 1 столбец, 4-й элемент)
    ax = figure.add_subplot(4, 1, 4) 
    
    pogr_styles = {'Явная': 'r-', 'Неявная': 'g--', 'Кранка-Николсона': 'b:'}
    
    max_pogr = 0
    for name, (t_pogr, pogr_vals) in all_pogr.items():
        if len(t_pogr) > 0:
            ax.plot(t_pogr, pogr_vals, pogr_styles.get(name, 'm-'), label=name)
            max_pogr = max(max_pogr, np.max(pogr_vals))

    ax.set_title("Погрешность (норма) в зависимости от времени для разных схем")
    ax.set_xlabel("t")
    ax.set_ylabel("||u_solved - u_true||_max")
    ax.set_ylim(0, max_pogr * 1.1)
    ax.grid(True, linestyle='--')
    ax.legend(loc='lower right', ncol=3)
    
    # Применяем tight_layout для 2D/Погрешности
    figure.tight_layout() 
    
    # Отображаем обе фигуры
    plt.show(block=True)
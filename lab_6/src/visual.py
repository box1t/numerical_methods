import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.lines import Line2D
from natsort import natsorted
from mpl_toolkits.mplot3d import Axes3D

# --- Вспомогательные функции (без изменений) ---

def _load_data_from_file(full_path):
    """Загружает данные x, u_solved, u_true и t_cur из одного файла."""
    x, u_solved, u_true = [], [], []
    t_cur = 0.0
    try:
        with open(full_path, "r") as f:
            lines = f.readlines()
            if lines:
                t_cur = float(lines[0].strip())
                data_array = np.loadtxt(lines[1:], dtype=float)
                if data_array.ndim == 1:
                     data_array = data_array.reshape(1, -1)
                
                x = data_array[:, 0]
                u_solved = data_array[:, 1]
                u_true = data_array[:, 2]

    except Exception as e:
        pass
    return np.array(x), np.array(u_solved), np.array(u_true), t_cur

def _load_data_for_scheme(path):
    """
    Загружает все x, u_solved, u_true для всех времен из папки.
    """
    solutions = {}
    
    data_files = [f for f in os.listdir(path) if f.endswith(".txt") and f[0] != 'p']
    data_files = natsorted(data_files)
    
    for file in data_files:
        x, u_solved, u_true, t_cur = _load_data_from_file(os.path.join(path, file))
        if len(x) > 0:
            solutions[t_cur] = (x, u_solved, u_true)
            
    pogr_path = os.path.join(path, 'p.txt')
    t_pogr, pogr_vals = [], []
    try:
        data = np.loadtxt(pogr_path)
        if data.ndim > 1:
            t_pogr = data[:, 0]
            pogr_vals = data[:, 1]
        elif data.size > 0: 
            t_pogr = [data[0]]
            pogr_vals = [data[1]]
    except FileNotFoundError:
         pass
    except Exception:
         pass
         
    return solutions, t_pogr, pogr_vals

# --- ГЛАВНАЯ ФУНКЦИЯ (С ИЗМЕНЕНИЯМИ) ---

def visualise_comparison(paths_dict: dict, max_t=5.0):
    """
    Визуализирует сравнение схем на одном 2D-графике (все T), 
    3D эволюцию u(x,t) и график погрешности (с линейной шкалой).
    """
    
    # --- 0. Константы и стили ---
    
    # 💡 ИЗМЕНЕНИЕ: Стиль для каждой комбинации схемы и порядка (Только маркеры и стили, без цвета)
    scheme_full_styles = {
        'Точное':             {'linestyle': '-', 'marker': None, 'lw': 2.5},
        'Явная':        {'linestyle': '-', 'marker': 'o', 'ms': 5, 'lw': 1.5},
        'Неявная':      {'linestyle': '-', 'marker': 's', 'ms': 5, 'lw': 1.5},
    }
    
    # Выбираем 3 момента времени T для 2D сравнения
    T_target_list = [max_t * 0.1, max_t * 0.5, max_t * 0.8] 
    
    # --- 1. Загрузка всех данных (оставляем без изменений) ---
    all_data = {}
    all_pogr = {}
    
    first_data_key = next(iter(paths_dict), None)
    if not first_data_key:
        print("Ошибка: Словарь путей пуст.")
        return

    for name, path in paths_dict.items():
        solutions, t_pogr, pogr_vals = _load_data_for_scheme(path)
        all_data[name] = solutions
        all_pogr[name] = (t_pogr, pogr_vals)
    
    all_times = natsorted(all_data[first_data_key].keys())
    
    chosen_data = {} 
    max_val_u = 0
    max_pogr_val = 0
    
    for T_target in T_target_list:
        if not all_times: continue
        closest_t_index = np.argmin(np.abs(np.array(all_times) - T_target))
        t_actual = all_times[closest_t_index]
        chosen_data[t_actual] = {}
        
        for name in paths_dict.keys():
             if t_actual in all_data[name]:
                x, u_solved, u_true = all_data[name][t_actual]
                chosen_data[t_actual][name] = (x, u_solved, u_true)
                max_val_u = max(max_val_u, np.max(np.abs(u_solved)))
        
        if t_actual in all_data[first_data_key]:
            x_true, _, u_true = all_data[first_data_key][t_actual]
            chosen_data[t_actual]['Точное'] = (x_true, u_true, u_true)
            max_val_u = max(max_val_u, np.max(np.abs(u_true)))
            
    for _, (_, pogr_vals) in all_pogr.items():
        if len(pogr_vals) > 0:
            max_pogr_val = max(max_pogr_val, np.max(pogr_vals))

    # --- 2. Создание фигуры для 2D Сравнения и Погрешности ---
    
    figure_2d_pogr = plt.figure(figsize=(18, 10)) 
    
    # ---------------------------------------------------------------------
    # 2.1. ОДИН 2D ГРАФИК СРАВНЕНИЯ (объединенный) - СТРОГИЙ СТИЛЬ
    # ---------------------------------------------------------------------
    
    ax_comparison = figure_2d_pogr.add_subplot(4, 1, (1, 3)) 
    ax_comparison.set_title("Сравнение схем в разные моменты времени T", fontsize=16)
    ax_comparison.set_xlabel("x")
    ax_comparison.set_ylabel("u(x, T)")
    ax_comparison.grid(True, linestyle='--', alpha=0.6)
    
    ax_comparison.set_ylim(-0.15, max_val_u * 1.1) 
    
    all_scheme_names = ['Точное'] + natsorted([k for k in paths_dict.keys()])
    
    # Используем контрастные цвета для T
    # 💡 ИЗМЕНЕНИЕ: Цвета для T: Оранжевый, Зеленый, Синий.
    time_colors_map = None
    time_colors = ['#FF7F0E', '#2CA02C', '#1F77B4'] # Orange, Green, Blue
    if len(chosen_data.keys()) > 3:
        # Если вдруг данных больше 3, используем палитру Dark2
        time_colors_map = plt.cm.get_cmap('Dark2', len(chosen_data.keys()))
        time_colors = [time_colors_map(i) for i in range(len(chosen_data.keys()))]
    
    scheme_handles_for_legend = {}
    
    # Отрисовка
    for t_idx, (t_actual, t_data) in enumerate(chosen_data.items()):
        
        # 💡 ИМЯ МЕТКИ ВРЕМЕНИ
        time_label_text = f"T$_{t_idx + 1}$"
        
        current_time_color = time_colors[t_idx] if t_idx < len(time_colors) else 'gray'
        
        # 1. Сначала Точное 
        if 'Точное' in t_data:
            x_true, u_true_vals, _ = t_data['Точное']
            style = scheme_full_styles['Точное']
            
            line, = ax_comparison.plot(x_true, u_true_vals, 
                               color='black', 
                               linestyle=style['linestyle'], 
                               linewidth=style['lw'], 
                               marker=style['marker'],
                               zorder=3)
            
            # Сохраняем handle для легенды (нейтральная подсветка)
            if 'Точное' not in scheme_handles_for_legend:
                 scheme_handles_for_legend['Точное'] = Line2D([], [], color='black', 
                                                                       linestyle=style['linestyle'], 
                                                                       linewidth=style['lw'], 
                                                                       marker=style['marker'], 
                                                                       label='Точное')
                 
        # 2. Затем численные схемы
        for name in [n for n in all_scheme_names if n != 'Точное' and n in t_data]:
            x_solved_vals, u_solved_vals, _ = t_data[name]
            style = scheme_full_styles.get(name)
            
            if style:
                line, = ax_comparison.plot(x_solved_vals, u_solved_vals, 
                                   color=current_time_color, 
                                   linestyle=style['linestyle'], 
                                   linewidth=style['lw'], 
                                   marker=style['marker'], 
                                   markersize=style['ms'], 
                                   fillstyle='none',
                                   zorder=2)
                
                # Сохраняем handle для легенды (нейтральная подсветка)
                if name not in scheme_handles_for_legend:
                     scheme_handles_for_legend[name] = Line2D([], [], color='black', 
                                                                 linestyle=style['linestyle'], 
                                                                 linewidth=style['lw'], 
                                                                 marker=style['marker'], 
                                                                 markersize=style['ms'], 
                                                                 fillstyle='none',
                                                                 label=name)
        
        # 💡 ДОБАВЛЕНИЕ ТЕКСТОВОЙ МЕТКИ T1, T2, T3
        if 'Точное' in t_data:
            x_true, u_true_vals, _ = t_data['Точное']
            
            # Берем значение u(x=pi, T) для Точного решения
            last_x_index = len(x_true) - 1
            u_at_pi = u_true_vals[last_x_index]
            x_at_pi = x_true[last_x_index]

            ax_comparison.text(x_at_pi + 0.05, u_at_pi, 
                                time_label_text, 
                                color=current_time_color, 
                                fontsize=14, 
                                fontweight='bold',
                                verticalalignment='center',
                                backgroundcolor='white',
                                bbox=dict(facecolor='white', alpha=0.6, boxstyle="round,pad=0.2"))
                                
    # --- Легенда ВРЕМЕНИ (T) ---
    # Добавляем T-метки в легенду
    time_handles = []
    for t_idx, t_actual in enumerate(chosen_data.keys()):
        t_color = time_colors[t_idx] if t_idx < len(time_colors) else 'gray'
        time_handles.append(Line2D([], [], color=t_color, 
                                   linestyle='-', 
                                   linewidth=2, 
                                   label=f'T$_{t_idx + 1}$ = {t_actual:.3f}'))
    
    # 💡 ФИНАЛЬНЫЙ ШАГ: Объединяем легенды в одну
    final_handles = list(scheme_handles_for_legend.values()) + time_handles
    
    ax_comparison.legend(handles=final_handles, loc='upper right', ncol=3, fontsize='small', framealpha=0.9, title='Схемы и Моменты времени')
    
    # ---------------------------------------------------------------------
    # 2.2. ГРАФИК ПОГРЕШНОСТИ (нижний подграфик) - Стили для погрешности
    # ---------------------------------------------------------------------
    
    ax_pogr = figure_2d_pogr.add_subplot(4, 1, 4) 
    ax_pogr.set_title("Погрешность (max norm) в зависимости от времени для разных схем", fontsize=12)
    ax_pogr.set_xlabel("t (Время)")
    ax_pogr.set_ylabel("||u_solved - u_true||_max")
    ax_pogr.grid(True, linestyle='--', alpha=0.6)
    
    ax_pogr.set_ylim(0, max_pogr_val * 1.5 if max_pogr_val > 0 else 0.05)
    
    pogr_handles = []
    
    # 💡 Цвета для погрешности (как на графике 1/2): Синий/Красный
    pogr_scheme_colors = {
        'Явная': 'blue',
        'Неявная': 'red',
    }

    for name, (t_pogr, pogr_vals) in all_pogr.items():
        if len(t_pogr) > 0 and name != 'Точное':
            style = scheme_full_styles.get(name)
            color = pogr_scheme_colors.get(name, 'gray')
            
            if style:
                # Используем стиль линии схемы, но цвет по типу Явная/Неявная
                line, = ax_pogr.plot(t_pogr, pogr_vals, 
                             color=color, 
                             linestyle=style['linestyle'], 
                             linewidth=style['lw'], 
                             label=name)
                pogr_handles.append(line)

    ax_pogr.legend(handles=pogr_handles, loc='upper right', ncol=4)
    
    figure_2d_pogr.tight_layout() 
    
    # ---------------------------------------------------------------------
    # 3. 3D ЭВОЛЮЦИЯ РЕШЕНИЯ u(x,t) (ОТДЕЛЬНОЕ ОКНО) - Без изменений
    # ---------------------------------------------------------------------
    
    scheme_names_for_3d = ['Точное'] + natsorted([k for k in paths_dict.keys()])[:2] 
    
    scheme_data_mapping = {
        'Точное': all_data[first_data_key],
    }
    for name, data in all_data.items():
        scheme_data_mapping[name] = data

    
    scheme_names_for_3d = [name for name in scheme_names_for_3d if scheme_data_mapping.get(name)]
    
    if len(scheme_names_for_3d) > 0:
        figure_3d = plt.figure(figsize=(18, 6))
    
        for idx, scheme_name in enumerate(scheme_names_for_3d):
            current_scheme_data = scheme_data_mapping.get(scheme_name)
            
            if current_scheme_data is None:
                 continue

            X_grid_list, T_grid_list, Z_grid_list = [], [], [] 
            
            for t_val in all_times:
                if t_val in current_scheme_data:
                    x_vals, u_solved_vals, u_true_vals = current_scheme_data[t_val]
                    
                    X_grid_list.append(x_vals)
                    T_grid_list.append(np.full_like(x_vals, t_val))
                    
                    Z_grid_list.append(u_true_vals if scheme_name == 'Точное' else u_solved_vals)
            
            if not X_grid_list:
                continue

            X_surf = np.array(X_grid_list)
            T_surf = np.array(T_grid_list)
            Z_surf = np.array(Z_grid_list)
            
            ax = figure_3d.add_subplot(1, len(scheme_names_for_3d), idx + 1, projection='3d')
            
            ax.view_init(elev=20, azim=-120)
            
            surf = ax.plot_surface(X_surf, T_surf, Z_surf, 
                                   cmap='viridis', 
                                   edgecolor='gray', 
                                   linewidth=0.2,
                                   rstride=2, cstride=2,
                                   antialiased=True
                                  )

            ax.set_title(f"{scheme_name}", fontsize=14)
            ax.set_ylabel("t")
            ax.set_xlabel("x")
            ax.set_zlabel("u")
            
            ax.set_zlim(-max_val_u * 3, max_val_u * 3)
            
            figure_3d.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='u')
        
        figure_3d.tight_layout()
    
    plt.show(block=True)
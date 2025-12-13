import os
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from natsort import natsorted

def visualise(path: str, title: str = "График", num_plots: int = 3):

    # --- Загрузка и выбор данных (без изменений) ---
    data_files = [file for file in os.listdir(path) if file.endswith(".txt") and file[0]!='p']
    data_files = natsorted(data_files)
    
    if len(data_files) < num_plots:
        chosen_files = data_files
        num_plots = len(data_files)
    else:
        data_step = (len(data_files)-1)//(num_plots-1)
        chosen_files = [data_files[data_step*i] for i in range(num_plots)]

    figure = plt.figure(figsize=(18, 9))
    counter = 1

    # --- Отрисовка 3D графиков (ИЗМЕНЕНИЯ ЗДЕСЬ) ---
    for file in chosen_files:
        full_path = os.path.join(path, file)

        with open(full_path, "r") as f:
            lines = f.readlines()
            cur_iter = int(lines[0])
            ys = []; xs = []; us = []; ts_list = []
            for line in lines[1:]:
                parts = line.strip().split()
                ys.append(float(parts[0]))
                xs.append(float(parts[1]))
                us.append(float(parts[2]))
                ts_list.append(float(parts[3]))

        unique_ys = sorted(set(ys))
        unique_xs = sorted(set(xs))
        ny = len(unique_ys)
        nx = len(unique_xs)
        Z_solved = np.zeros((ny, nx))
        Z_true = np.zeros((ny, nx))
        y_to_idx = {y: j for j, y in enumerate(unique_ys)}
        x_to_idx = {x: i for i, x in enumerate(unique_xs)}
        for k in range(len(ys)):
            j = y_to_idx[ys[k]]
            i = x_to_idx[xs[k]]
            Z_solved[j, i] = us[k]
            Z_true[j, i] = ts_list[k]

        XX, YY = np.meshgrid(unique_xs, unique_ys)

        ax = figure.add_subplot(2, len(chosen_files), counter, projection='3d')
        
        # 1. Аналитическое решение (более явный цвет, чуть выше прозрачность)
        ax.plot_surface(XX, YY, Z_true, 
                        cmap='summer',         # Зеленый градиент
                        zorder = 1, 
                        alpha=0.6,             # Умеренная прозрачность
                        linewidth=0, 
                        antialiased=True
                       )

        # 2. Вычисленное решение (яркий плазменный градиент)
        ax.plot_surface(XX, YY, Z_solved, 
                        cmap='plasma',         # Яркий плазменный градиент
                        zorder = 10, 
                        alpha=0.9,             # Высокая непрозрачность для яркости
                        linewidth=0, 
                        antialiased=True
                       )
        
        #ax.set_title("Эволюция погрешности решения от итерации", fontsize=14)
        ax.set_title(f"{title}, iter={cur_iter}")
        #ax.set_title(title +", iter=" + str(cur_iter))

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u')

        # Обновленная легенда
        # Берем средние цвета из соответствующих цветовых карт для легенды
        legend_elements = [
            Patch(facecolor=plt.cm.plasma(0.5), alpha=0.9, label='Вычисленное решение'), 
            Patch(facecolor=plt.cm.summer(0.5), alpha=0.6, label='Аналитическое решение')
        ]
        ax.legend(handles=legend_elements)

        counter += 1

    # --- Отрисовка графика погрешности ---
    pogr_path = path + '/p.txt'
    x = []; pogr = []
    
    try:
        with open(pogr_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                line_stripped = line.strip().split(' ')
                if len(line_stripped) == 2:
                    x.append(float(line_stripped[0]))
                    pogr.append(float(line_stripped[1]))
    except FileNotFoundError:
        print(f"Файл погрешности не найден: {pogr_path}")
        return

    # Размещаем график погрешности в нижней части фигуры
    p = figure.add_subplot(2, 1, 2) 
    
    p.plot(x, pogr, 
           marker='o',             
           markersize=5,           
           linestyle='-',          
           linewidth=2.0,          
           color='#1F77B4',        
           label='Погрешность (Max Norm)') 

    p.set_title("Эволюция погрешности решения от итерации", fontsize=14)
    p.set_xlabel('Итерация (k)')
    p.set_ylabel(r'$||u_{solved}^k - u_{true}||_{\infty}$') 
    p.grid(True, linestyle='--', alpha=0.6) 
    p.legend(loc='upper right', framealpha=0.9)
    # ----------------------------------------------------

    plt.tight_layout()
    plt.show(block=True)
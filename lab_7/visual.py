import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import os
from natsort import natsorted

def visualise(path: str, title: str = "График", num_plots: int = 3):


    data_files = [file for file in os.listdir(path) if file.endswith(".txt") and file[0]!='p']

    data_files = natsorted(data_files)

    data_step = (len(data_files)-1)//(num_plots-1)

    chosen_files = [data_files[data_step*i] for i in range(num_plots)]

    figure = plt.figure(figsize=(18, 9))
    counter = 1
    
    for file in chosen_files:
        full_path = os.path.join(path, file)

        with open(full_path, "r") as f:
            lines = f.readlines()
            cur_iter = int(lines[0])
            ys = []
            xs = []
            us = []
            ts_list = []
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
        ax.plot_wireframe(XX, YY, Z_true, color='black', zorder = 1)
        ax.plot_surface(XX, YY, Z_solved, cmap='viridis', zorder = 10, alpha=0.6, linewidth=0, antialiased=True)
        
        
        ax.set_title(title +", iter=" + str(cur_iter))

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u')

        legend_elements = [
            Patch(facecolor='blue', label='solved'),
            Line2D([0], [0], color='black', label='true')
        ]
        ax.legend(handles=legend_elements)

        counter += 1

    pogr_path = path + '/p.txt'
    with open(pogr_path, "r") as f:
        lines = f.readlines()
        x = []; pogr = []
        for line in lines:
            line_stripped = line.strip().split(' ')
            x.append(float(line_stripped[0]))
            pogr.append(float(line_stripped[1]))

        p = figure.add_subplot(2,1,2)

        p.plot(x, pogr, marker='', linestyle='-', color='red')

        plt.title("Погрешность в зависимости от итерации")
        plt.xlabel('iter')
        plt.ylabel('norm')
        plt.grid()

    plt.tight_layout()
    plt.show(block=True)
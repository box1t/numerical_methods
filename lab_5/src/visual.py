import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from natsort import natsorted

DATA_FOLDER = "results"
PATH = os.path.join(os.path.split(os.path.realpath(__file__))[0], DATA_FOLDER)

def visualise(path):

    # ------- Загружаем файлы решения -------
    data_files = [file for file in os.listdir(path) if file.endswith(".txt") and file[0] != 'p']
    data_files = natsorted(data_files)

    # выбираем 3 файла по аналогии с твоим старым кодом
    chosen_files = [
        data_files[len(data_files)//4],
        data_files[len(data_files)//2],
        data_files[-1]
    ]

    figure = plt.figure(figsize=(18, 12))

    # =====================================================================
    #                           ---- 2D ГРАФИКИ ----
    # =====================================================================

    counter = 1
    for file in chosen_files:
        full_path = os.path.join(path, file)

        x = []
        u_solved = []
        u_true = []

        with open(full_path, "r") as f:
            lines = f.readlines()
            t_str = lines[0]

            for line in lines[1:]:
                parts = line.strip().split()
                x.append(float(parts[0]))
                u_solved.append(float(parts[1]))
                u_true.append(float(parts[2]))

        ax = figure.add_subplot(3, len(chosen_files), counter)
        ax.plot(x, u_solved, 'r-', label='solved')
        ax.plot(x, u_true, 'b--', label='true')

        ax.set_title(f"2D график, t={t_str}")
        ax.set_xlabel("x")
        ax.set_ylabel("u")
        ax.grid()
        ax.legend()

        counter += 1

    # =====================================================================
    #                           ---- 3D ГРАФИКИ ----
    # =====================================================================

    for file in chosen_files:
        full_path = os.path.join(path, file)

        x = []
        u_solved = []
        u_true = []

        with open(full_path, "r") as f:
            lines = f.readlines()
            cur_t = float(lines[0])

            for line in lines[1:]:
                parts = line.strip().split()
                x.append(float(parts[0]))
                u_solved.append(float(parts[1]))
                u_true.append(float(parts[2]))

        x = np.array(x)

        # ---- создаём "толщину" поверхности ----
        ny = 25
        y = np.linspace(-0.05, 0.05, ny)

        XX, YY = np.meshgrid(x, y)

        # дублируем решение вдоль y
        Z_solved = np.tile(u_solved, (ny, 1))
        Z_true   = np.tile(u_true,   (ny, 1))

        ax = figure.add_subplot(3, len(chosen_files), counter, projection='3d')

        # true wireframe (мелкая сетка)
        ax.plot_wireframe(XX, YY, Z_true, color='black', linewidth=0.6)

        # solved surface (из разных слоёв)
        ax.plot_surface(XX, YY, Z_solved, cmap='viridis', alpha=0.8, linewidth=0)

        ax.set_title(f"3D график, t={cur_t:.3f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("u")

        legend_elements = [
            Patch(facecolor='blue', label='solved'),
            Line2D([0], [0], color='black', label='true')
        ]
        ax.legend(handles=legend_elements)

        counter += 1


    # =====================================================================
    #                           ---- ГРАФИК ПОГРЕШНОСТИ ----
    # =====================================================================

    pogr_path = os.path.join(path, 'p.txt')

    with open(pogr_path, "r") as f:
        lines = f.readlines()
        t = []
        pogr = []
        for line in lines:
            parts = line.strip().split()
            t.append(float(parts[0]))
            pogr.append(float(parts[1]))

    ax = figure.add_subplot(3, 1, 3)
    ax.plot(t, pogr, 'r-')

    ax.set_title("Погрешность в зависимости от времени")
    ax.set_xlabel("t")
    ax.set_ylabel("norm")
    ax.grid()

    plt.tight_layout()
    plt.show(block=True)

# visualise(PATH)

import os
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from natsort import natsorted

def visualise(path: str, title: str = "График", num_plots: int = 3, t_round: int = 3):

    data_files = [file for file in os.listdir(path) if file.endswith(".txt") and file[0] != 'p']
    data_files = natsorted(data_files)

    data_step = (len(data_files) - 1) // (num_plots - 1)
    chosen_files = [data_files[data_step * i] for i in range(num_plots)]

    figure = plt.figure(figsize=(20, 10))
    counter = 1

    # =====================================================================
    #                           ---- 2D ГРАФИКИ ----
    # =====================================================================

    for file in chosen_files:
        full_path = os.path.join(path, file)

        x = []
        u_solved = []
        u_true = []

        with open(full_path, "r") as f:
            lines = f.readlines()
            t_cur = float(lines[0])

            for line in lines[1:]:
                parts = line.strip().split()
                x.append(float(parts[0]))
                u_solved.append(float(parts[1]))
                u_true.append(float(parts[2]))

        p = figure.add_subplot(3, num_plots, counter)

        p.plot(x, u_solved, marker='o', linestyle='-', color='red', label='solved')
        p.plot(x, u_true, marker='o', linestyle='--', color='blue', label='true')

        p.set_title(title + ", t=" + str(round(t_cur, t_round)))
        p.set_xlabel('x')
        p.set_ylabel('u')
        p.grid()
        p.legend()

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
            t_cur = float(lines[0])

            for line in lines[1:]:
                parts = line.strip().split()
                x.append(float(parts[0]))
                u_solved.append(float(parts[1]))
                u_true.append(float(parts[2]))

        x = np.array(x)

        # создаём "толщину" поверхности
        ny = 25
        y = np.linspace(-0.05, 0.05, ny)

        XX, YY = np.meshgrid(x, y)

        Z_solved = np.tile(u_solved, (ny, 1))
        Z_true = np.tile(u_true, (ny, 1))

        ax = figure.add_subplot(3, num_plots, counter, projection='3d')

        # true wireframe
        ax.plot_wireframe(XX, YY, Z_true, color='black', linewidth=0.7)

        # solved surface
        ax.plot_surface(XX, YY, Z_solved, cmap='viridis', alpha=0.8, linewidth=0)

        ax.set_title(f"3D, t={round(t_cur, t_round)}")
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
    #                        ---- ГРАФИК ПОГРЕШНОСТИ ----
    # =====================================================================

    pogr_path = os.path.join(path, 'p.txt')

    with open(pogr_path, "r") as f:
        lines = f.readlines()
        t_vals = []
        pogr_vals = []
        for line in lines:
            parts = line.strip().split()
            t_vals.append(float(parts[0]))
            pogr_vals.append(float(parts[1]))

    p = figure.add_subplot(3, 1, 3)

    p.plot(t_vals, pogr_vals, color='red')
    p.set_title("Погрешность в зависимости от времени")
    p.set_xlabel("t")
    p.set_ylabel("norm")
    p.grid()

    plt.tight_layout()
    plt.show(block=True)


# import matplotlib.pyplot as plt
# import os
# from natsort import natsorted

# # DATA_FOLDER = "results"
# # PATH = os.path.join(os.path.split(os.path.realpath(__file__))[0], DATA_FOLDER)

# def visualise(path: str, title: str = "График", num_plots: int = 3, t_round: int = 3):

#     data_files = [file for file in os.listdir(path) if file.endswith(".txt") and file[0]!='p']

#     data_files = natsorted(data_files)

#     data_step = (len(data_files)-1)//(num_plots-1)

#     chosen_files = [data_files[data_step*i] for i in range(num_plots)]

#     figure = plt.figure(figsize=(18, 9))
#     counter = 1

#     for file in chosen_files:
#         full_path = os.path.join(path, file)

#         x = []; u_solved = []; u_true = []
#         with open(full_path, "r") as f:
#             lines = f.readlines()
#             t_cur = float(lines[0])
#             for line in lines[1:]:
#                 line_stripped = line.strip().split(' ')
#                 x.append(float(line_stripped[0]))
#                 u_solved.append(float(line_stripped[1]))
#                 u_true.append(float(line_stripped[2]))
            
#             p = figure.add_subplot(2, len(chosen_files), counter)
#             #plt.ylim(None, 1.0)

#             p.plot(x, u_solved, marker='o', linestyle='-', color='red', label='solved')
#             p.plot(x, u_true, marker='o', linestyle='--', color='blue', label='true')

#             plt.title(title +", t=" + str(round(t_cur, t_round)))
#             plt.xlabel('x')
#             plt.ylabel('u')
#             plt.grid()
#             plt.legend()

#         counter += 1

#     pogr_path = path + '/p.txt'
#     with open(pogr_path, "r") as f:
#         lines = f.readlines()
#         x = []; pogr = []
#         for line in lines:
#             line_stripped = line.strip().split(' ')
#             x.append(float(line_stripped[0]))
#             pogr.append(float(line_stripped[1]))

#         p = figure.add_subplot(2,1,2)
#         #plt.ylim(None, 0.01)

#         p.plot(x, pogr, marker='', linestyle='-', color='red')

#         plt.title("Погрешность в зависимости от времени")
#         plt.xlabel('x')
#         plt.ylabel('norm')
#         plt.grid()

#     plt.tight_layout()
#     plt.show(block=True)

# # visualise(PATH)
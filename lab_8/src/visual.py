import os
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Patch
from natsort import natsorted

def visualise(path: str, title: str = "График", num_plots: int = 3, round_t: int = 3):

    # --- Загрузка и выбор данных ---
    data_files = [file for file in os.listdir(path)
                  if file.endswith(".txt") and file[0] != 'p']
    data_files = natsorted(data_files)

    if len(data_files) < num_plots:
        chosen_files = data_files
        num_plots = len(data_files)
    else:
        data_step = (len(data_files) - 1) // (num_plots - 1)
        chosen_files = [data_files[data_step * i] for i in range(num_plots)]

    figure = plt.figure(figsize=(18, 9))
    counter = 1

    # --- 3D графики решений ---
    for file in chosen_files:
        full_path = os.path.join(path, file)

        with open(full_path, "r") as f:
            lines = f.readlines()
            cur_t = float(lines[0])

            ys, xs, us, ts_list = [], [], [], []
            for line in lines[1:]:
                y, x, u, t_val = map(float, line.strip().split())
                ys.append(y)
                xs.append(x)
                us.append(u)
                ts_list.append(t_val)

        unique_ys = sorted(set(ys))
        unique_xs = sorted(set(xs))

        ny, nx = len(unique_ys), len(unique_xs)
        Z_solved = np.zeros((ny, nx))
        Z_true   = np.zeros((ny, nx))

        y_to_idx = {y: j for j, y in enumerate(unique_ys)}
        x_to_idx = {x: i for i, x in enumerate(unique_xs)}

        for k in range(len(ys)):
            j = y_to_idx[ys[k]]
            i = x_to_idx[xs[k]]
            Z_solved[j, i] = us[k]
            Z_true[j, i]   = ts_list[k]

        XX, YY = np.meshgrid(unique_xs, unique_ys)

        ax = figure.add_subplot(2, len(chosen_files), counter, projection='3d')

        # --- Аналитическое решение ---
        ax.plot_surface(
            XX, YY, Z_true,
            cmap='summer',
            alpha=0.6,
            zorder=1,
            linewidth=0,
            antialiased=True
        )

        # --- Численное решение ---
        ax.plot_surface(
            XX, YY, Z_solved,
            cmap='plasma',
            alpha=0.9,
            zorder=10,
            linewidth=0,
            antialiased=True
        )

        ax.set_title(f"{title}, t={round(cur_t, round_t)}")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u')

        legend_elements = [
            Patch(facecolor=plt.cm.plasma(0.5), alpha=0.9,
                  label='Вычисленное решение'),
            Patch(facecolor=plt.cm.summer(0.5), alpha=0.6,
                  label='Аналитическое решение')
        ]
        ax.legend(handles=legend_elements)

        counter += 1

    # --- График погрешности ---
    pogr_path = os.path.join(path, 'p.txt')
    x, pogr = [], []

    try:
        with open(pogr_path, "r") as f:
            for line in f:
                t_val, err = map(float, line.strip().split())
                x.append(t_val)
                pogr.append(err)
    except FileNotFoundError:
        print(f"Файл погрешности не найден: {pogr_path}")
        return

    p = figure.add_subplot(2, 1, 2)
    p.plot(
        x, pogr,
        marker='o',
        markersize=5,
        linestyle='-',
        linewidth=2.0,
        label='Погрешность (Max Norm)'
    )

    p.set_title("Эволюция погрешности решения во времени", fontsize=14)
    p.set_xlabel('t')
    p.set_ylabel(r'$||u_{solved}(t) - u_{true}(t)||_{\infty}$')
    p.grid(True, linestyle='--', alpha=0.6)
    p.legend(loc='upper right', framealpha=0.9)

    plt.tight_layout()
    plt.show(block=True)

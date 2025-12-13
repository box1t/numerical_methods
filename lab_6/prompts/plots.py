

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from natsort import natsorted

# DATA_FOLDER = "results"
# PATH = os.path.join(os.path.split(os.path.realpath(__file__))[0], DATA_FOLDER)

def visualise(path: str, title: str = "График", num_plots: int = 3, t_round: int = 3):

    data_files = [file for file in os.listdir(path) if file.endswith(".txt") and file[0]!='p']

    data_files = natsorted(data_files)

    data_step = (len(data_files)-1)//(num_plots-1)

    chosen_files = [data_files[data_step*i] for i in range(num_plots)]

    figure = plt.figure(figsize=(18, 9))
    counter = 1

    for file in chosen_files:
        full_path = os.path.join(path, file)

        x = []; u_solved = []; u_true = []
        with open(full_path, "r") as f:
            lines = f.readlines()
            t_cur = float(lines[0])
            for line in lines[1:]:
                line_stripped = line.strip().split(' ')
                x.append(float(line_stripped[0]))
                u_solved.append(float(line_stripped[1]))
                u_true.append(float(line_stripped[2]))
            
            p = figure.add_subplot(2, len(chosen_files), counter)
            #plt.ylim(None, 1.0)

            p.plot(x, u_solved, marker='o', linestyle='-', color='red', label='solved')
            p.plot(x, u_true, marker='o', linestyle='--', color='blue', label='true')

            plt.title(title +", t=" + str(round(t_cur, t_round)))
            plt.xlabel('x')
            plt.ylabel('u')
            plt.grid()
            plt.legend()

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
        #plt.ylim(None, 0.01)

        p.plot(x, pogr, marker='', linestyle='-', color='red')

        plt.title("Погрешность в зависимости от времени")
        plt.xlabel('t')
        plt.ylabel('norm')
        plt.grid()

    plt.tight_layout()
    plt.show(block=True)

def fixed_scale_animate(path: str, title: str = "График", interval: int = 30, t_round: int = 3, 
                       xlim: tuple = None, ylim: tuple = None, save_path: str = None, fps: int = 10):
    """
    Версия с фиксированным масштабом для сравнения амплитуд
    """
    
    data_files = [file for file in os.listdir(path) if file.endswith(".txt") and file[0] != 'p']
    data_files = natsorted(data_files)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Если границы не заданы, вычисляем из всех данных
    if xlim is None or ylim is None:
        all_x = []
        all_u = []
        for file in data_files:
            full_path = os.path.join(path, file)
            with open(full_path, "r") as f:
                lines = f.readlines()
                for line in lines[1:]:
                    vals = line.strip().split()
                    all_x.append(float(vals[0]))
                    all_u.append(float(vals[1]))
                    all_u.append(float(vals[2]))
        
        if xlim is None:
            x_margin = (max(all_x) - min(all_x)) * 0.05
            xlim = (min(all_x) - x_margin, max(all_x) + x_margin)
        
        if ylim is None:
            u_margin = (max(all_u) - min(all_u)) * 0.1
            ylim = (min(all_u) - u_margin, max(all_u) + u_margin)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    # Подготовка данных
    all_data = []
    for file in data_files:
        full_path = os.path.join(path, file)
        with open(full_path, "r") as f:
            lines = f.readlines()
            x, u_solved, u_true = [], [], []
            for line in lines[1:]:
                vals = line.strip().split()
                x.append(float(vals[0]))
                u_solved.append(float(vals[1]))
                u_true.append(float(vals[2]))
            
            all_data.append((x, u_solved, u_true))
    
    # Создание линий
    line_solved, = ax.plot([], [], 'ro-', label='solved', markersize=4, linewidth=2)
    line_true, = ax.plot([], [], 'b--', label='true', linewidth=2, alpha=0.8)
    
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    def animate_frame(i):
        x, u_solved, u_true = all_data[i]
        line_solved.set_data(x, u_solved)
        line_true.set_data(x, u_true)
        ax.set_title(f"{title}")
        return line_solved, line_true
    
    anim = animation.FuncAnimation(
        fig, animate_frame, frames=len(all_data),
        interval=interval, blit=True, repeat=True
    )
    
    plt.tight_layout()
    plt.show()
    return anim
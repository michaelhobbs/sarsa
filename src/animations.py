import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import os


def makeThumbnail(data, title, out, maze=None):
    l, = plt.plot([], [], '.', markersize=40)
    l2, = plt.plot([], [], 'r')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot([0.1], [0.1], 'gx')
    circle1 = plt.Circle((0.8, 0.8), 0.1, color='r', fill=False, label='Goal')
    plt.gcf().gca().add_artist(circle1)
    plt.title(title)
    plt.text(0.1, 0.9, f'step: {data.shape[1]}')
    plt.gca().set_aspect('equal', adjustable='box')
    l.set_data(data[:, -1])
    l2.set_data(data[:, :])
    if maze is not None:
        plt.plot(maze[:, :, 0], maze[:, :, 1], 'k',  markersize=10)

    # To save the animation, use the command: line_ani.save('lines.mp4')
    plt.savefig(f'{out}.png')
    plt.close()


def animate(data, title, out, interval=100, maze=None):
    ''' animate experiment for every set of coordinates in data array
    save animation to out file path'''
    #text = tx.Text(0.1, 0.9, 'testing')

    fig = plt.figure(figsize=[6, 6])

    l, = plt.plot([], [], '.', markersize=40)
    l2, = plt.plot([], [], 'r')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot([0.1], [0.1], 'gx')
    circle1 = plt.Circle((0.8, 0.8), 0.1, color='r', fill=False, label='Goal')
    plt.gcf().gca().add_artist(circle1)
    plt.title(title)
    text = plt.text(0.1, 0.9, '')
    plt.gca().set_aspect('equal', adjustable='box')
    if maze is not None:
        plt.plot(maze)

    def update_line(num, data, line, line2):
        line.set_data(data[:, num])
        line2.set_data(data[:, :num])
        #line.axes.text(0.1, 0.9, f'step: {num}')
        text.set_text(f'step: {num}')
        return line,
    line_ani = animation.FuncAnimation(fig, update_line, data.shape[1], fargs=(data, l, l2),
                                       interval=interval, blit=True)

    # To save the animation, use the command: line_ani.save('lines.mp4')
    line_ani.save(f'{out}.mp4')
    plt.close()
    makeThumbnail(data, title, out)
    # return HTML(line_ani.to_jshtml())


def generateAnimation(epsilon):
    ''' generate animation for report '''
    data = np.loadtxt(
        f'output/epsilons/{epsilon}/2/ratCoords/ratCoords_50.csv', delimiter=",")
    data = data.reshape(int(len(data)/2), 2).T
    title = f'Epsilon: {epsilon}'
    out = f'images/animations/last_trial_{epsilon}'
    interval = (100, 30)[epsilon in ['0.5', '0.9']]
    print('interval: ' + str(interval))
    animate(data, title, out, interval)


#os.makedirs('images/animations', exist_ok=True)
# generateAnimation('0.1')
# _ = [generateAnimation(i) for i in ['0.1', '0.5', '0.9',
#                                    'logDecay', 'linearDecay1-0', 'expDecay']]

for i in range(100):
    walls = np.array([[[0.5, 0], [0.5, 0.5]], [[0.5, 0.5], [0.2, 0.5]]])
    data = np.loadtxt(
        f'output/maze/lin/{i}/ratCoords/ratCoords_500.csv', delimiter=",")
    data = data.reshape(int(len(data)/2), 2).T
    title = f'Maze'
    out = f'images/animations/mega_maze/maze_50_{i}'
    makeThumbnail(data, title, out, maze=walls)

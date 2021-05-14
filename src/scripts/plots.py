import numpy as np
import matplotlib.pyplot as plt


def plotQuiverWeights(weights, trial, experimentNumber, xAxisSteps, outputDirectory):
    axisSize = xAxisSteps
    xDelta = 1 / (axisSize - 1)
    axis = [(j % xAxisSteps) * xDelta for j in range(axisSize)]

    x = axis
    y = axis

    X, Y = np.meshgrid(x, y)

    maxElements = [np.amax(weights[i, :]) for i in range(axisSize*axisSize)]
    maxElement = np.amax(maxElements)
    maxIndexes = [np.random.choice(np.where(weights[i, :] == maxElements[i])[
                                   0]) for i in range(axisSize*axisSize)]

    def convertIndexToX(i):
        return np.array([0, 0.7, 1, 0.7, 0, -0.7, -1, -0.7])[i]

    def convertIndexToY(i):
        return np.array([-1, -0.7, 0, 0.7, 1, 0.7, 0, -0.7])[i]
    u = [convertIndexToX(maxIndexes[:])]
    v = [convertIndexToY(maxIndexes[:])]
    _, ax = plt.subplots(figsize=(9, 9))
    rectangle = plt.Rectangle((0, 0), 1, 1, ec='blue', fc='none')
    plt.gca().add_patch(rectangle)
    circle2 = plt.Circle((0.8, 0.8), 0.1, color='r', fill=False)
    ax.add_artist(circle2)
    plt.plot([0.1], [0.1], 'rx')
    ax.quiver(X, Y, u, v)
    # ax.quiver(X,Y,u,v, scale=2*axisSize, scale_units='xy')
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.axis([-0.1, 1.1, -0.1, 1.1])
    ax.set_aspect('equal')
    plt.savefig(outputDirectory + str(experimentNumber) +
                '/plots/trial_'+str(trial)+'.png')
    plt.close()

    u = np.multiply([convertIndexToX(maxIndexes[:])],
                    (maxElements / maxElement))
    v = np.multiply([convertIndexToY(maxIndexes[:])],
                    (maxElements / maxElement))
    _, ax = plt.subplots(figsize=(9, 9))
    rectangle2 = plt.Rectangle((0, 0), 1, 1, ec='blue', fc='none')
    plt.gca().add_patch(rectangle2)
    plt.plot([0.1], [0.1], 'rx')
    ax.quiver(X, Y, u, v)
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.axis([-0.1, 1.1, -0.1, 1.1])
    ax.set_aspect('equal')
    circle1 = plt.Circle((0.8, 0.8), 0.1, color='r', fill=False)
    plt.gcf().gca().add_artist(circle1)
    plt.savefig(outputDirectory + str(experimentNumber) +
                '/plots/trial_'+str(trial)+'_norm.png')
    plt.close()

    _, ax = plt.subplots(figsize=(9, 9))
    rectangle2 = plt.Rectangle((0, 0), 1, 1, ec='blue', fc='none')
    plt.gca().add_patch(rectangle2)
    plt.plot([0.1], [0.1], 'rx')
    ax.quiver(X, Y, u, v, scale=2*axisSize, scale_units='xy')
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.axis([-0.1, 1.1, -0.1, 1.1])
    ax.set_aspect('equal')
    circle1 = plt.Circle((0.8, 0.8), 0.1, color='r', fill=False)
    plt.gcf().gca().add_artist(circle1)
    plt.savefig(outputDirectory + str(experimentNumber) +
                '/plots/trial_'+str(trial)+'_full_norm.png')
    plt.close()


def historyHistogram(history):
    plt.hist2d(history[:, 0], history[:, 1], 20)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar(label='Number of time-steps')
    plt.title('Time spent in Cells of Grid')
    plt.xlabel('x coordinate')
    plt.ylabel('y coordinate')
    plt.savefig('images/historyHistogram.png')


def history(history):
    plt.figure(figsize=(5, 5))
    plt.plot(history[:, 0], history[:, 1], '.')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('State History')
    plt.xlabel('x coordinate')
    plt.ylabel('y coordinate')
    plt.savefig('history.png')


def rewardPie(rewards):
    plt.pie([np.sum(rewards == 0), np.sum(rewards == -2), np.sum(rewards == 10)])
    successRate = np.round(np.sum(rewards == 10) / len(rewards) * 100, 2)
    neutralRate = np.round(np.sum(rewards == 0) / len(rewards) * 100, 2)
    failRate = np.round(np.sum(rewards == -2) / len(rewards) * 100, 2)
    plt.legend([f'neutral = {neutralRate}%',
                f'negative = {failRate}%', f'positive = {successRate}%'])
    plt.title('Time spent (%) by type of state')
    plt.savefig('timeSpentPie.png')
    plt.show()


def firstMoves(firstMoves):
    plt.subplots(1, 1)
    plt.annotate(str(0), (firstMoves[0, 0], firstMoves[0, 1]-0.005),
                 clip_on=True, ha='center', in_layout=True, va='center', color='black')
    plt.annotate(str(
        1), (firstMoves[1, 0]+0.005, firstMoves[1, 1]-0.005), ha='center', va='center')
    plt.annotate(str(2), (firstMoves[2, 0]+0.005,
                          firstMoves[2, 1]), ha='center', va='center')
    plt.annotate(str(
        3), (firstMoves[3, 0]+0.005, firstMoves[3, 1]+0.005), ha='center', va='center')
    plt.annotate(
        str(4), (firstMoves[4, 0], firstMoves[4, 1]+0.005), ha='center', va='center')
    plt.annotate(str(
        5), (firstMoves[5, 0]-0.005, firstMoves[5, 1]+0.005), ha='center', va='center')
    plt.annotate(str(6), (firstMoves[6, 0]-0.005,
                          firstMoves[6, 1]), ha='center', va='center')
    plt.annotate(str(
        7), (firstMoves[7, 0]-0.005, firstMoves[7, 1]-0.005), ha='center', va='center')
    for color in range(8):
        plt.arrow(0.1, 0.1, 0.1-firstMoves[color, 0], 0.1-firstMoves[color, 1],
                  length_includes_head=True, width=0.0005, ls='--', lw=0.001)
    plt.axis([0.025, 0.175, 0.025, 0.175])
    plt.plot([0.1], [0.1], 'k.')
    plt.xlabel('x')
    plt.ylabel('y  ', rotation=0)
    plt.title('Agent\'s first available moves')
    plt.legend(['Starting position'])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

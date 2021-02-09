
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

import numpy as np
import math


class Grid:
    '''
    Models the State space in which the agent acts.
    The space is 2-dimensional and square.

    Parameters:
    - resolution (x,y) tuple: number of grid cells per axis
    - functions to detectCollision, detectGoal, returnAgentToGrid,

    Overriding an instance of this class's functions allows to run QL on a different problem.
    Creating a new Grid needs to re-use the dimensions and number of directions of the previous Grid in order to be compatible with the existing QLearn model.
    '''

    def __init__(self,
                 xAxisSteps=20,
                 yAxisSteps=20,
                 minCoords=[0, 0],
                 maxCoords=[1, 1],
                 numDirections=8,
                 l=0.03,
                 initCoords=[0.1, 0.1]):
        self._xAxisSteps = xAxisSteps
        self._yAxisSteps = yAxisSteps
        self._minCoords = minCoords
        self._maxCoords = maxCoords
        self._placeCellCoords = self.generateCoordsArray()
        self._numDirections = numDirections
        self.l = l
        diagonalDelta = math.sqrt(l**2/2)
        self.positionDeltas = np.array([[0, -l], [diagonalDelta, -diagonalDelta], [l, 0], [diagonalDelta, diagonalDelta], [
            0, l], [-diagonalDelta, diagonalDelta], [-l, 0], [-diagonalDelta, -diagonalDelta]])
        self.initCoords = initCoords

    def cellCoordsByIndex(self, index, xDelta, yDelta):
        '''
        Converts an index of flattened 2d space to [x, y] coordinates.
        The first cell has index 0 and coordinates [0, 0].
        The last cell has index ``numCells`` and coordinates ``[1, 1]``.

        args: \n
            index : number in [0, (numCells - 1)]
            xDelta : distance between 2 cells along the x-axis
            yDelta : distance between 2 cells along the y-axis

        returns: [x, y] coordinate tuple for the centre of the i-th place cell
        '''
        return np.array([(index % self._xAxisSteps) * xDelta, (math.floor(index / self._xAxisSteps) * yDelta)])

    def generateCoordsArray(self):
        '''
        Generates coordinate array for the grid using a linear scale.
        '''
        numPlaceCells = self._xAxisSteps * self._yAxisSteps
        xDelta = 1 / (self._xAxisSteps - 1)
        yDelta = 1 / (self._yAxisSteps - 1)
        return np.array([self.cellCoordsByIndex(j, xDelta, yDelta) for j in range(numPlaceCells)])

    def updatePosition(self, currentCoords, direction):
        reward = 0
        newPosition = currentCoords + self.positionDeltas[direction]
        hasCollision = self.detectCollision(newPosition)
        hasGoal = self.detectGoal(newPosition)
        if (hasCollision):
            newPosition = self.returnRatToGrid(newPosition)
            reward -= 2
        elif (hasGoal):
            reward += 10
        return [newPosition, reward, hasGoal]

    def detectCollision(self, coords):
        '''
        Tests if a point is within the grid.
        Equivalant to checking if a state should be rewarded negatively.
        '''
        m = self._minCoords
        M = self._maxCoords
        return coords[0] < m[0] or coords[1] < m[1] or coords[0] > M[0] or coords[1] > M[1]

    def detectGoal(self, coords):
        '''
        Tests if a point is within the goal area.
        '''
        goalCenter = [0.8, 0.8]
        goalRadius = 0.1
        distFromCenter = np.sqrt((coords[0]-goalCenter[0])**2 +
                                 (coords[1]-goalCenter[1])**2)
        return distFromCenter <= goalRadius

    def returnRatToGrid(self, coords):
        '''
        Given a point outside the grid, return it to the grid.
        '''
        return np.minimum(np.maximum(self._minCoords, coords), self._maxCoords)

    def getCellIndex(self, coords):
        return math.floor(round(coords[0]*(self._xAxisSteps-1), 0) + round(coords[1]*(self._yAxisSteps-1), 0) * self._xAxisSteps)

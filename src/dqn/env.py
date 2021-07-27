import numpy as np
import math


class Grid:
    '''
    Models the State/action space in which the agent acts.
    The space is 2-dimensional and square.

    reset() -> reset to episode initial state
    step(action) -> state_next, reward, done
    '''

    def __init__(self,
                 minCoords=[0, 0],
                 maxCoords=[1, 1],
                 numDirections=8,
                 l=0.03,
                 initCoords=[0.1, 0.1]):
        self._minCoords = minCoords
        self._maxCoords = maxCoords
        self._numDirections = numDirections
        self.l = l
        diagonalDelta = math.sqrt(l**2/2)
        self.positionDeltas = np.array([[0, -l], [diagonalDelta, -diagonalDelta], [l, 0], [diagonalDelta, diagonalDelta], [
            0, l], [-diagonalDelta, diagonalDelta], [-l, 0], [-diagonalDelta, -diagonalDelta]])
        self.initCoords = initCoords
        self.state = initCoords

    def reset(self):
        self.state = self.initCoords
        return self.state

    def step(self, direction):
        reward = -0.01
        newPosition = self.state + self.positionDeltas[direction]
        hasCollision = self.detectCollision(newPosition)
        hasGoal = self.detectGoal(newPosition)
        if (hasCollision):
            newPosition = self.returnRatToGrid(newPosition)
            reward = -0.2
        elif (hasGoal):
            reward = 1
        self.state = newPosition
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

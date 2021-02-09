import numpy as np
import math


class QLearn:
    '''
    Manages Q-Learning model:
    - state space
    - agent state
    - agent actions
    - eligibility trace
    - weights
    - model parameters
    '''

    def __init__(self,
                 experimentNumber,
                 outputDirectory='output/',
                 xAxisSteps=20,
                 yAxisSteps=20,
                 learningRate=0.005,
                 rewardDiscountRate=0.95,
                 eligibilityTraceDecayRate=0.95,
                 numDirections=8,
                 minCoords=[0, 0],
                 maxCoords=[1, 1],
                 l=0.03,
                 weights=None):
        self.outputDirectory = outputDirectory + str(experimentNumber)
        self.xAxisSteps = xAxisSteps
        self.yAxisSteps = yAxisSteps
        self.learningRate = learningRate
        self.rewardDiscountRate = rewardDiscountRate
        self.eligibilityTraceDecayRate = eligibilityTraceDecayRate
        self.placeCellCoords = self.generateCoordsArray()
        self.numDirections = 8
        self.minCoords = [0, 0]
        self.maxCoords = [1, 1]
        self.l = l
        diagonalDelta = math.sqrt(l**2/2)
        self.positionDeltas = np.array([[0, -l], [diagonalDelta, -diagonalDelta], [l, 0], [diagonalDelta, diagonalDelta], [
            0, l], [-diagonalDelta, diagonalDelta], [-l, 0], [-diagonalDelta, -diagonalDelta]])
        self.trialNumber = 0
        self.weights = (weights, self.initWeights())[weights is None]

    def initWeights(self):
        return 0.001 * np.array(np.random.rand(self.xAxisSteps * self.yAxisSteps, self.numDirections))

    def cellCoordsByIndex(self, index, xDelta, yDelta):
        '''
        Converts an index of flattened 2d space to [x, y] coordinates.
        The first cell has index 0 and coordinates [0, 0].
        The last cell has index _numCells_ and coordinates [1, 1].

        args: \n
                index : number in [0, (numCells - 1)]
                xDelta : distance between 2 cells along the x-axis
                yDelta : distance between 2 cells along the y-axis

        returns: [x, y] coordinate tuple for the centre of the i-th place cell
        '''
        return np.array([(index % self.xAxisSteps) * xDelta, (math.floor(index / self.xAxisSteps) * yDelta)])

    def calculatePlaceCellActivity(self, ratCoords):
        sigma = 0.05
        denom = 2 * sigma**2
        num = -((self.placeCellCoords[:, 0]-ratCoords[0]) **
                2 + (self.placeCellCoords[:, 1]-ratCoords[1])**2)
        return np.exp(num / denom)

    def generateCoordsArray(self):
        numPlaceCells = self.xAxisSteps * self.yAxisSteps
        xDelta = 1 / (self.xAxisSteps - 1)
        yDelta = 1 / (self.yAxisSteps - 1)
        return np.array([self.cellCoordsByIndex(j, xDelta, yDelta) for j in range(numPlaceCells)])

    def calculateOutputNeuronActivityForDirection(self, ratCoords, direction):
        return np.sum(np.multiply(self.weights[:, direction], self.calculatePlaceCellActivity(ratCoords)))

    def calculateOutputNeuronActivity(self, ratCoords):
        return [self.calculateOutputNeuronActivityForDirection(ratCoords, a) for a in range(self.numDirections)]

    def selectDirection(self, epsilon, ratCoords):
        randomNumber = np.random.rand(1)
        if (randomNumber < epsilon):
            return self.selectRandomDirection()
        else:
            outputNeuronActivity = self.calculateOutputNeuronActivity(
                ratCoords)
            arr = np.array(outputNeuronActivity)
            maxElement = np.amax(arr)
            maxIndexes = np.where(arr == maxElement)
            # pick random index when several share maxValue
            randomFromMaxIndexes = np.random.choice(maxIndexes[0])
            return randomFromMaxIndexes

    def selectRandomDirection(self):
        return np.random.randint(0, self.numDirections)

    def detectCollision(self, coords):
        m = self.minCoords
        M = self.maxCoords
        return coords[0] < m[0] or coords[1] < m[1] or coords[0] > M[0] or coords[1] > M[1]

    def detectGoal(self, coords):
        return np.sqrt((coords[0]-0.8)**2 + (coords[1]-0.8)**2) <= 0.1

    def updatePosition(self, currentCoords, direction):
        return currentCoords + self.positionDeltas[direction]

    def returnRatToGrid(self, coords):
        return np.minimum(np.maximum(self.minCoords, coords), self.maxCoords)

    def getCellIndex(self, coords):
        return math.floor(round(coords[0]*(self.xAxisSteps-1), 0) + round(coords[1]*(self.yAxisSteps-1), 0) * self.xAxisSteps)

    def updateEligibilityTrace(self, currentCoords, direction, eligibilityTrace):
        updated = eligibilityTrace
        # TODO: we can use r(s) instead of 1 // or rather we should use r(s)
        updated[self.getCellIndex(currentCoords), direction] += 1
        return updated

    def decayEligibilityTrace(self, eligibilityTrace):
        return eligibilityTrace * self.rewardDiscountRate * self.eligibilityTraceDecayRate

    def updateWeights(self, delta, eligibilityTrace):
        self.weights += (self.learningRate * delta * eligibilityTrace)

    def runSingleTimeStep(self, currentPosition, currentDirection, eligibilityTrace, epsilon):
        reward = 0
        newPosition = self.updatePosition(
            currentPosition, currentDirection)  # s'
        # by taking action a given state s -> s'
        hasCollision = self.detectCollision(newPosition)
        # by taking action a given state s -> s'
        hasGoal = self.detectGoal(newPosition)
        if (hasCollision):
            newPosition = self.returnRatToGrid(newPosition)
            reward -= 2
        elif (hasGoal):
            reward += 10
        # update eligibility trace at ratCoords before move for chosen direction
        eligibilityTrace = self.updateEligibilityTrace(
            currentPosition, currentDirection, eligibilityTrace)

        # calculate next action a' which will be taken in next iteration and will be used to update the model
        nextDirection = self.selectDirection(epsilon, newPosition)
        anticipatedOutputNeuronActivityInNextStep = self.calculateOutputNeuronActivityForDirection(
            newPosition, nextDirection)  # np.amax(calculateOutputNeuronActivity(newPosition, weights))
        # calculate delta and update weights
        # delta = r - Q(s,a) + gamma*Q(s',a')
        delta = reward - self.calculateOutputNeuronActivityForDirection(
            currentPosition, currentDirection) + self.rewardDiscountRate * anticipatedOutputNeuronActivityInNextStep
        self.updateWeights(delta, eligibilityTrace)
        eligibilityTrace = self.decayEligibilityTrace(eligibilityTrace)
        newDirection = self.selectDirection(
            epsilon, newPosition)  # next action a'
        return [newPosition, newDirection, reward, hasGoal, eligibilityTrace]

    def runTrial(self, epsilon):
        '''
        Runs a trial on the model after resetting the agent state.

        '''
        # initialization
        # ratCoords is initial position s
        ratCoords = [0.1, 0.1]
        ratCoordsHistory = np.array([])
        ratCoordsHistory = np.append(ratCoordsHistory, ratCoords)
        currentDirection = self.selectDirection(
            epsilon, ratCoords)  # initial action a
        directionHistory = np.array([])
        directionHistory = np.append(directionHistory, currentDirection)
        trialReward = 0
        maxSteps = 10000
        currentStep = 0
        hasGoal = False
        self.trialNumber += 1

        eligibilityTrace = np.zeros(
            (self.xAxisSteps * self.yAxisSteps, self.numDirections))

        while (currentStep < maxSteps and not hasGoal):
            [ratCoords, currentDirection, reward, hasGoal, eligibilityTrace] = self.runSingleTimeStep(
                ratCoords, currentDirection, eligibilityTrace, epsilon)  # then run again on output of this function (s', a')
            ratCoordsHistory = np.append(ratCoordsHistory, ratCoords)
            directionHistory = np.append(directionHistory, currentDirection)
            trialReward += reward
            currentStep += 1

        print('Reached Goal:', hasGoal)
        print('Steps:', currentStep)
        print('Trial Reward:', trialReward)
        # print ('Final Position:', ratCoords)

        # self.save(ratCoordsHistory, 'ratCoords_' +  str(self.trialNumber), 'ratCoords')
        # self.save(directionHistory, 'directions_' + str(self.trialNumber), 'directions')
        return [eligibilityTrace, hasGoal, currentStep, trialReward, ratCoords]

    def save(self, data, fileName, directory):
        np.savetxt(self.outputDirectory + '/' +
                   directory + '/' + str(fileName) + '.csv', data, delimiter=",")


class Grid:
    '''
    Models the State and Action space in which the agent acts
    The space is 2-dimensional and square.
    Parameters:
    - resolution (x,y) tuple: number of grid cells per axis
    - stepSize float: distance of agent's single step
    - functions to detectCollision, detectGoal, returnAgentToGrid, 
    '''

    def __init__(self,
                 xAxisSteps=20,
                 yAxisSteps=20,
                 numDirections=8,
                 minCoords=[0, 0],
                 maxCoords=[1, 1],
                 l=0.03):
        self.xAxisSteps = xAxisSteps
        self.yAxisSteps = yAxisSteps
        self.l = l

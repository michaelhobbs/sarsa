import numpy as np
import os

from . import grid


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
                 learningRate=0.005,
                 rewardDiscountRate=0.95,
                 eligibilityTraceDecayRate=0.95,
                 maxSteps=10000,
                 weights=None,
                 onPolicy=True):
        self.outputDirectory = outputDirectory + str(experimentNumber)
        self.learningRate = learningRate
        self.rewardDiscountRate = rewardDiscountRate
        self.eligibilityTraceDecayRate = eligibilityTraceDecayRate
        self.maxSteps = maxSteps
        self.grid = grid.Grid()
        self.trialNumber = 0
        self.weights = (weights, self.initWeights())[weights is None]
        self.onPolicy = onPolicy
        os.makedirs(outputDirectory, exist_ok=True)

    def initWeights(self):
        return 0.001 * np.array(np.random.rand(self.grid._xAxisSteps * self.grid._yAxisSteps, self.grid._numDirections))

    def calculatePlaceCellActivity(self, ratCoords):
        sigma = 0.05
        denom = 2 * sigma**2
        num = -((self.grid._placeCellCoords[:, 0]-ratCoords[0]) **
                2 + (self.grid._placeCellCoords[:, 1]-ratCoords[1])**2)
        return np.exp(num / denom)

    def calculateOutputNeuronActivityForDirection(self, ratCoords, direction, activity=None):
        if activity is None:
            activity = self.calculatePlaceCellActivity(ratCoords)
        return np.sum(np.multiply(self.weights[:, direction], activity))

    def calculateOutputNeuronActivity(self, ratCoords):
        activity = self.calculatePlaceCellActivity(ratCoords)
        return [self.calculateOutputNeuronActivityForDirection(ratCoords, a, activity) for a in range(self.grid._numDirections)]

    def selectDirection(self, epsilon, ratCoords):
        # TODO: return the Q array for the selected direction, to avoid recalculating it later in the algo
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
        return np.random.randint(0, self.grid._numDirections)

    def updateEligibilityTrace(self, currentCoords, direction, eligibilityTrace):
        eligibilityTrace[:,
                         direction] += self.calculatePlaceCellActivity(currentCoords)
        return eligibilityTrace

    def decayEligibilityTrace(self, eligibilityTrace):
        return eligibilityTrace * self.rewardDiscountRate * self.eligibilityTraceDecayRate

    def updateWeights(self, delta, eligibilityTrace):
        self.weights += self.learningRate * delta * eligibilityTrace

    def runSingleTimeStep(self, currentPosition, currentDirection, eligibilityTrace, epsilon):
        [newPosition, reward, hasGoal] = self.grid.updatePosition(
            currentPosition, currentDirection)  # s'
        # update eligibility trace at ratCoords before move for chosen direction
        eligibilityTrace = self.updateEligibilityTrace(
            currentPosition, currentDirection, eligibilityTrace)

        anticipatedOutputNeuronActivityInNextStep = 0
        newDirection = 0
        if self.onPolicy:
            # calculate next action a' which will be taken in next iteration and will be used to update the model
            newDirection = self.selectDirection(epsilon, newPosition)
            anticipatedOutputNeuronActivityInNextStep = self.calculateOutputNeuronActivityForDirection(
                newPosition, newDirection)
        else:
            anticipatedOutputNeuronActivityInNextStep = np.max(
                self.calculateOutputNeuronActivity(newPosition))

        # calculate delta and update weights
        delta = reward - self.calculateOutputNeuronActivityForDirection(
            currentPosition, currentDirection) + self.rewardDiscountRate * anticipatedOutputNeuronActivityInNextStep

        self.updateWeights(delta, eligibilityTrace)
        eligibilityTrace = self.decayEligibilityTrace(eligibilityTrace)
        if not self.onPolicy:
            newDirection = self.selectDirection(
                epsilon, newPosition)
        return [newPosition, newDirection, reward, hasGoal, eligibilityTrace]

    def runTrial(self, epsilon):
        '''
        Runs a trial on the model after resetting the agent state.

        '''
        # initialization
        # ratCoords is initial position s
        ratCoords = self.grid.initCoords
        ratCoordsHistory = np.array([ratCoords])
        currentDirection = self.selectDirection(
            epsilon, ratCoords)  # initial action a
        directionHistory = np.array([currentDirection])
        trialReward = 0
        currentStep = 0
        hasGoal = False
        self.trialNumber += 1

        eligibilityTrace = np.zeros(
            (self.grid._xAxisSteps * self.grid._yAxisSteps, self.grid._numDirections))

        while (currentStep < self.maxSteps and not hasGoal):
            [ratCoords, currentDirection, reward, hasGoal, eligibilityTrace] = self.runSingleTimeStep(
                ratCoords, currentDirection, eligibilityTrace, epsilon)  # then run again on output of this function (s', a')
            ratCoordsHistory = np.append(ratCoordsHistory, ratCoords)
            directionHistory = np.append(directionHistory, currentDirection)
            trialReward += reward
            currentStep += 1

        if self.trialNumber % 5 == 1 or self.trialNumber == 50:
            print('Reached Goal:', hasGoal)
            print('Steps:', currentStep)
            print('Trial Reward:', trialReward)
            print('Final Position:', ratCoords)

        self.save(ratCoordsHistory, 'ratCoords_' +
                  str(self.trialNumber), 'ratCoords')
        self.save(directionHistory, 'directions_' +
                  str(self.trialNumber), 'directions')
        # self.saveWeights(str(self.trialNumber))
        return [eligibilityTrace, hasGoal, currentStep, trialReward, ratCoords]

    def saveWeights(self, fileName):
        saveDir = self.outputDirectory + '/weights/'
        os.makedirs(saveDir, exist_ok=True)
        np.savetxt(saveDir + str(fileName) + '.csv',
                   self.weights, delimiter=",")

    def save(self, data, fileName, directory):
        saveDir = self.outputDirectory + '/' + directory + '/'
        os.makedirs(saveDir, exist_ok=True)
        np.savetxt(saveDir + str(fileName) + '.csv', data, delimiter=",")

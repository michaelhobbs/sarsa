import math
import os

import numpy as np
import plots

outputDirectory = 'output/'
xAxisSteps = 20
yAxisSteps = 20

learningRate = 0.005
rewardDiscountRate = 0.95
eligibilityTraceDecayRate = 0.95


def cellCoordsByIndex(index, xAxisSteps, yAxisSteps, xDelta, yDelta):
    '''
    Converts an index of flattened 2d space to [x, y] coordinates.
    The first cell has index 0 and coordinates [0, 0].
    The last cell has index _numCells_ and coordinates [1, 1].

    args: \n
            index : number in [0, (numCells - 1)]
            xAxisSteps : number of columns
            yAxisSteps : number of rows
            xDelta : distance between 2 cells along the x-axis
            yDelta : distance between 2 cells along the y-axis

    returns: [x, y] coordinate tuple for the centre of the i-th place cell
    '''
    return np.array([(index % xAxisSteps) * xDelta, (math.floor(index / xAxisSteps) * yDelta)])


# TODO: vectorize to use np.exp and no for loop
def calculatePlaceCellActivity(ratCoords, placeCellCoords):
    sigma = 0.05
    return np.array([math.exp(- ((j[0]-ratCoords[0])**2 + (j[1]-ratCoords[1])**2) / (2 * sigma**2)) for j in placeCellCoords])


def generateCoordsArray(xAxisSteps, yAxisSteps):
    numPlaceCells = xAxisSteps * yAxisSteps
    xDelta = 1 / (xAxisSteps - 1)
    yDelta = 1 / (yAxisSteps - 1)
    return np.array([cellCoordsByIndex(j, xAxisSteps, yAxisSteps, xDelta, yDelta) for j in range(numPlaceCells)])


placeCellCoords = generateCoordsArray(xAxisSteps, yAxisSteps)

numDirections = 8

minCoords = [0, 0]
maxCoords = [1, 1]

l = 0.03
diagonalDelta = math.sqrt(l**2/2)
positionDeltas = np.array([[0, -l], [diagonalDelta, -diagonalDelta], [l, 0], [diagonalDelta, diagonalDelta], [
                          0, l], [-diagonalDelta, diagonalDelta], [-l, 0], [-diagonalDelta, -diagonalDelta]])

# Q(s, a)


def calculateOutputNeuronActivityForDirection(ratCoords, direction, weights):
    return np.sum(np.multiply(weights[:, direction], calculatePlaceCellActivity(ratCoords, placeCellCoords)))


def calculateOutputNeuronActivity(ratCoords, weights):
    return [calculateOutputNeuronActivityForDirection(ratCoords, a, weights) for a in range(numDirections)]


def selectDirection(epsilon, ratCoords, weights):
    randomNumber = np.random.rand(1)
    if (randomNumber < epsilon):
        return selectRandomDirection()
    else:
        outputNeuronActivity = calculateOutputNeuronActivity(
            ratCoords, weights)
        arr = np.array(outputNeuronActivity)
        maxElement = np.amax(arr)
        maxIndexes = np.where(arr == maxElement)
        # pick random index when several share maxValue
        randomFromMaxIndexes = np.random.choice(maxIndexes[0])
        return randomFromMaxIndexes


def selectRandomDirection():
    randomDirection = np.random.randint(0, numDirections)
    return randomDirection


def detectCollision(coords):
    return coords[0] < minCoords[0] or coords[1] < minCoords[1] or coords[0] > maxCoords[0] or coords[1] > maxCoords[1]


def detectGoal(coords):
    return np.sqrt((coords[0]-0.8)**2 + (coords[1]-0.8)**2) <= 0.1


def updatePosition(currentCoords, direction):
    return currentCoords + positionDeltas[direction]


def returnRatToGrid(coords):
    return np.minimum(np.maximum(minCoords, coords), maxCoords)


def getCellIndex(coords):
    return math.floor(round(coords[0]*(xAxisSteps-1), 0) + round(coords[1]*(yAxisSteps-1), 0) * xAxisSteps)


def updateEligibilityTrace(currentCoords, direction, eligibilityTrace):
    updated = eligibilityTrace
    # TODO: we can use r(s) instead of 1
    updated[getCellIndex(currentCoords), direction] += 1
    return updated


def decayEligibilityTrace(eligibilityTrace):
    return eligibilityTrace * rewardDiscountRate * eligibilityTraceDecayRate


def updateWeights(delta, weights, eligibilityTrace):
    return weights + (learningRate * delta * eligibilityTrace)


def runSingleTimeStep(currentPosition, currentDirection, weights, eligibilityTrace, epsilon):
    reward = 0
    newPosition = updatePosition(currentPosition, currentDirection)  # s'
    # by taking action a given state s -> s'
    hasCollision = detectCollision(newPosition)
    hasGoal = detectGoal(newPosition)  # by taking action a given state s -> s'
    if (hasCollision):
        newPosition = returnRatToGrid(newPosition)
        reward -= 2
    elif (hasGoal):
        reward += 10
    # update eligibility trace at ratCoords before move for chosen direction
    eligibilityTrace = updateEligibilityTrace(
        currentPosition, currentDirection, eligibilityTrace)

    # calculate next action a' which will be taken in next iteration and will be used to update the model
    nextDirection = selectDirection(epsilon, newPosition, weights)
    anticipatedOutputNeuronActivityInNextStep = calculateOutputNeuronActivityForDirection(
        newPosition, nextDirection, weights)  # np.amax(calculateOutputNeuronActivity(newPosition, weights))
    # calculate delta and update weights
    # delta = r - Q(s,a) + gamma*Q(s',a')
    delta = reward - calculateOutputNeuronActivityForDirection(
        currentPosition, currentDirection, weights) + rewardDiscountRate * anticipatedOutputNeuronActivityInNextStep
    weights = updateWeights(delta, weights, eligibilityTrace)
    eligibilityTrace = decayEligibilityTrace(eligibilityTrace)
    newDirection = selectDirection(
        epsilon, newPosition, weights)  # next action a'
    return [newPosition, newDirection, reward, hasGoal, weights, eligibilityTrace]


def runTrial(weights, eligibilityTrace, epsilon, trialNumber, experimentNumber):
    # initialization
    # ratCoords is initial position s
    ratCoords = [0.1, 0.1]
    ratCoordsHistory = np.array([])
    ratCoordsHistory = np.append(ratCoordsHistory, ratCoords)
    currentDirection = selectDirection(
        epsilon, ratCoords, weights)  # initial action a
    directionHistory = np.array([])
    directionHistory = np.append(directionHistory, currentDirection)
    trialReward = 0
    maxSteps = 10000
    currentStep = 0
    hasGoal = False

    while (currentStep < maxSteps and not hasGoal):
        [ratCoords, currentDirection, reward, hasGoal, weights, eligibilityTrace] = runSingleTimeStep(
            ratCoords, currentDirection, weights, eligibilityTrace, epsilon)  # then run again on output of this function (s', a')
        ratCoordsHistory = np.append(ratCoordsHistory, ratCoords)
        directionHistory = np.append(directionHistory, currentDirection)
        trialReward += reward
        currentStep += 1

    print('Reached Goal:', hasGoal)
    print('Steps:', currentStep)
    print('Trial Reward:', trialReward)
    # print ('Final Position:', ratCoords)

    save(ratCoordsHistory, 'ratCoords_' +
         str(trialNumber), experimentNumber, 'ratCoords')
    save(directionHistory, 'directions_' +
         str(trialNumber), experimentNumber, 'directions')
    return [weights, eligibilityTrace, hasGoal, currentStep, trialReward, ratCoords]


def save(data, fileName, experimentNumber, directory):
    np.savetxt(outputDirectory + str(experimentNumber) + '/' +
               directory + '/' + str(fileName) + '.csv', data, delimiter=",")


def runExperiment(experimentNumber):
    os.makedirs(outputDirectory + str(experimentNumber) +
                '/plots', exist_ok=True)
    os.makedirs(outputDirectory + str(experimentNumber) +
                '/weights', exist_ok=True)
    os.makedirs(outputDirectory + str(experimentNumber) +
                '/eligibilityTrace', exist_ok=True)
    os.makedirs(outputDirectory + str(experimentNumber) +
                '/ratCoords', exist_ok=True)
    os.makedirs(outputDirectory + str(experimentNumber) +
                '/directions', exist_ok=True)
    maxTrials = 50
    currentTrial = 0
    epsilon = 0.9
    epsilonLinearDecay = (0.5 - 0.01) / 50
    weights = 0.001 * \
        np.array(np.random.rand(xAxisSteps * yAxisSteps, numDirections))
    plots.plotQuiverWeights(weights, -1, experimentNumber,
                            xAxisSteps, outputDirectory)
    trialRewardHistory = np.array([])
    trialGoalHistory = np.array([])
    trialDurationHistory = np.array([])
    finalPositionHistory = np.array([])
    while (currentTrial < maxTrials):
        print('[[START OF TRIAL]]:', currentTrial)
        eligibilityTrace = np.zeros((xAxisSteps * yAxisSteps, numDirections))
        [weights, eligibilityTrace, hasGoal, currentStep, trialReward, ratCoords] = runTrial(
            weights, eligibilityTrace, epsilon, currentTrial, experimentNumber)
        trialRewardHistory = np.append(trialRewardHistory, trialReward)
        trialGoalHistory = np.append(trialGoalHistory, hasGoal)
        trialDurationHistory = np.append(trialDurationHistory, currentStep)
        finalPositionHistory = np.append(finalPositionHistory, ratCoords)
        plots.plotQuiverWeights(weights, currentTrial,
                                experimentNumber, xAxisSteps, outputDirectory)
        save(weights, 'weights_' + str(currentTrial), experimentNumber, 'weights')
        save(eligibilityTrace, 'eligibilityTrace_' +
             str(currentTrial), experimentNumber, 'eligibilityTrace')
        epsilon = epsilon - epsilonLinearDecay  # max((0.95 * epsilon), 0.05)
        currentTrial += 1

    # print ('Reached Goal History:', trialGoalHistory)
    print('Number of Steps History:', trialDurationHistory)
    print('Trial Reward History:', trialRewardHistory)
    # print ('Final Position History:', finalPositionHistory)

    a = np.asarray(
        [trialGoalHistory, trialDurationHistory, trialRewardHistory])
    np.savetxt(outputDirectory + str(experimentNumber) +
               "/testTrial.csv", a, delimiter=",")

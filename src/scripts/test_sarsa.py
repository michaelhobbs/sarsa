from sarsa import getCellIndex, detectGoal, detectCollision, generateCoordsArray, returnRatToGrid, calculatePlaceCellActivity, calculateOutputNeuronActivityForDirection, calculateOutputNeuronActivity, cellCoordsByIndex
import numpy as np
import math


def test_returnRatToGrid():
    assert list(returnRatToGrid([0, 0])) == list([0, 0])
    assert list(returnRatToGrid([-0.001, 0])) == list([0, 0])
    assert list(returnRatToGrid([-0.001, 1])) == list([0, 1])
    assert list(returnRatToGrid([-0.001, 1.001])) == list([0, 1])
    assert list(returnRatToGrid([-0.001, 0.9])) == list([0, 0.9])
    assert list(returnRatToGrid([0.5, 1.1])) == list([0.5, 1])


def test_getCellIndex():
    assert getCellIndex([0, 0]) == 0
    assert getCellIndex([1, 1]) == (20*20-1)
    assert getCellIndex([0.05, 0]) == 1
    assert getCellIndex([0, 0.05]) == 20
    assert getCellIndex(cellCoordsByIndex(20, 20, 20, 1/19, 1/19)) == 20
    assert getCellIndex(cellCoordsByIndex(21, 20, 20, 1/19, 1/19)) == 21
    assert getCellIndex(cellCoordsByIndex(350, 20, 20, 1/19, 1/19)) == 350


def test_calculateOutputNeuronActivityForDirection():
    cellActivityMin = math.exp(-1/0.05**2)
    cellActivityMax = 1
    # direction 1 from first cell has weight 1, all others are 0
    # all cells contribute nothing when weight is 0
    # only the weight for the cell 0 contributes
    # a cell's contribution is 1 when the agent is on the cells center (exp(0) = 1)
    weights = np.zeros((20 * 20, 8))
    weights[0, 1] = 1
    assert calculateOutputNeuronActivityForDirection(
        [0, 0], 1, weights) == cellActivityMax
    # in this test case, only the furthest point on the grid contributes (weight 1)
    weights = np.zeros((20 * 20, 8))
    weights[399, 1] = 1
    assert calculateOutputNeuronActivityForDirection(
        [0, 0], 1, weights) == cellActivityMin
    # in this test case, both the point the agen t is on and the furthest point on the grid contribute (weight 1)
    weights = np.zeros((20 * 20, 8))
    weights[0, 1] = 1
    weights[399, 1] = 1
    assert calculateOutputNeuronActivityForDirection(
        [0, 0], 1, weights) == cellActivityMax + cellActivityMin


def test_calculatePlaceCellActivity():
    placeCellCoords = generateCoordsArray(20, 20)
    assert calculatePlaceCellActivity([0, 0], placeCellCoords)[0] == 1
    assert calculatePlaceCellActivity([1, 1], placeCellCoords)[
        0] == math.exp(-1/(0.05**2))
    assert calculatePlaceCellActivity([0.5, 0.5], placeCellCoords)[
        0] == math.exp(-0.5/(2*(0.05**2)))


def test_detectCollision():
    assert detectCollision([-0.001, 0.43]) == True
    assert detectCollision([0.343, -0.0001]) == True
    assert detectCollision([1.001, 0.43]) == True
    assert detectCollision([0.2, 1.001]) == True
    assert detectCollision([1.001, 1.001]) == True
    assert detectCollision([0.001, 0.001]) == False
    assert detectCollision([0, 0]) == False
    assert detectCollision([0, 1]) == False
    assert detectCollision([1, 0]) == False
    assert detectCollision([1, 1]) == False


def test_detectGoal():
    assert detectGoal([0.8, 0.8]) == True
    assert detectGoal([0.5, 0.5]) == False
    assert detectGoal([0.9, 0.8]) == True
    assert detectGoal([0.9001, 0.8]) == False

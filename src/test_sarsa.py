from sarsa import getCellIndex, detectGoal, detectCollision, generateCoordsArray, returnRatToGrid, calculatePlaceCellActivity, calculateOutputNeuronActivityForDirection, calculateOutputNeuronActivity, cellCoordsByIndex
import numpy as np
import math

def test_returnRatToGrid():
    assert list(returnRatToGrid([0,0])) == list([0,0])
    assert list(returnRatToGrid([-0.001,0])) == list([0,0])
    assert list(returnRatToGrid([-0.001,1])) == list([0,1])
    assert list(returnRatToGrid([-0.001,1.001])) == list([0,1])
    assert list(returnRatToGrid([-0.001,0.9])) == list([0,0.9])
    assert list(returnRatToGrid([0.5,1.1])) == list([0.5,1])
	
def test_getCellIndex():
    assert getCellIndex([0,0]) == 0
    assert getCellIndex([1,1]) == (20*20-1)
    assert getCellIndex([0.05,0]) == 1
    assert getCellIndex([0,0.05]) == 20
    assert getCellIndex(cellCoordsByIndex(20,20,20,1/19,1/19)) == 20
    assert getCellIndex(cellCoordsByIndex(21,20,20,1/19,1/19)) == 21
    assert getCellIndex(cellCoordsByIndex(350,20,20,1/19,1/19)) == 350

def test_calculateOutputNeuronActivityForDirection():
    weights = np.ones((20 * 20, 8))
    weights[:,1] = 2*np.ones((20 * 20))
    print(calculateOutputNeuronActivity([0,0], weights))
    assert calculateOutputNeuronActivityForDirection([0,0],2, weights) == 1

def test_calculatePlaceCellActivity():
    placeCellCoords = generateCoordsArray(20, 20)
    assert calculatePlaceCellActivity([0,0], placeCellCoords)[0] == 1
    assert calculatePlaceCellActivity([1,1], placeCellCoords)[0] == math.exp(-1/(0.05**2))
    assert calculatePlaceCellActivity([0.5,0.5], placeCellCoords)[0] == math.exp(-0.5/(2*(0.05**2)))

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
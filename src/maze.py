import numpy as np
from classes.ql.ql import QLearn
from classes.ql.grid import Grid


walls = (((0.5, 0), (0.5, 0.5)), ((0.5, 0.5), (0.2, 0.5)))
maze = Grid()


def intersect(p1, p2, p3, p4):
    '''intersection between line(p1, p2) and line(p3, p4)'''
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
    if denom == 0:  # parallel
        return None
    ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
    if ua < 0 or ua > 1:  # out of range
        return None
    ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / denom
    if ub < 0 or ub > 1:  # out of range
        return None
    x = x1 + ua * (x2-x1)
    y = y1 + ua * (y2-y1)
    return (x, y)


def mazeUpdate(state, action):
    ''' given s and a, return s', a', and r '''
    s = maze
    reward = 0
    newPosition = state + s.positionDeltas[action]
    hit = False
    for wall in walls:
        if intersect(state, newPosition, wall[0], wall[1]):
            hit = True
            break
    newPosition = (newPosition, state)[hit]
    hasCollision = s.detectCollision(newPosition)
    newPosition = np.clip(newPosition, 0, 1)
    hasGoal = s.detectGoal(newPosition)
    if (hit or hasCollision):
        reward = -1
    elif (hasGoal):
        reward = 100
    return [newPosition, reward, hasGoal]


maze.updatePosition = mazeUpdate


n_agents = 100
n_trials = 500
numSteps = np.zeros((n_agents, n_trials))
rewards = np.zeros((n_agents, n_trials))
outputDirectory = 'output/maze/lin/'
for n in range(n_agents):
    print('----------------RAT: [' + str(n) + ']')
    agent = QLearn(n,
                   outputDirectory=outputDirectory,
                   maxSteps=100000)
    agent.grid = maze
    agent.saveWeights('0')
    for t in range(n_trials):
        [eligibilityTrace, hasGoal, currentStep,
            trialReward, ratCoords] = agent.runTrial(1-1/(n_trials - 1)*t)
        numSteps[n, t] = currentStep
        rewards[n, t] = trialReward
results = {'epsilon': 'lin',
           'numSteps': numSteps, 'rewards': rewards, 'comments': 'linear decay from 1 to 0'}

file = f'{outputDirectory}results.npy'
np.save(file, results)
print(results)

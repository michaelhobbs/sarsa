# %%
import math
import os
import numpy as np
from ql.ql import QLearn

SEED = 1337
np.random.seed(1337)

# rats = 10
# for i in range(rats):
#     print('----------------RAT: [' + str(i) + ']')
#     QL = ql.QLearn(i, maxSteps=10000)
#     for j in range(50):
#         QL.runTrial(1 - j*1/50)

#     print('max weight: ' + str(np.max(QL.weights)))
#     print('mean weight: ' + str(np.mean(QL.weights)))
#     print('median weight: ' + str(np.median(QL.weights)))

# experiments
# //effect of epsilon [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]
# //epsilon 0 with zero weights init vs random low weights init vs rand 0 to 1 init
# //decaying epsilon: linear decay, log decay, exponential decay
# //removal of eligibility trace
# //off-policy
# distribution of weights, max weight, median, min, histogram, weights per direction in 3D

# %%
# theoretical best:
# √(0.7²+0.7²)−0.1 = 0.8899 (0.1 to 0.8 minus the circle radius of goal area)
# 0.8899 / 0.03 = 29.66 (distance to closest point of goal area by mvt speed)
epsilons = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]
results = []
seed = SEED
for epsilon in epsilons:
    seed += 1
    np.random.seed(seed)
    print('----------------EPSILON: [' + str(epsilon) + ']')
    numSteps = np.zeros((10, 50))
    rewards = np.zeros((10, 50))
    rats = 10
    for i in range(rats):
        print('----------------RAT: [' + str(i) + ']')
        QL = QLearn(i,
                    outputDirectory='output/epsilons/'+str(epsilon)+'/')
        QL.saveWeights('0')
        for j in range(50):
            [eligibilityTrace, hasGoal, currentStep,
                trialReward, ratCoords] = QL.runTrial(epsilon)
            numSteps[i, j] = currentStep
            rewards[i, j] = trialReward
    results = [*results, {'epsilon': epsilon,
                          'numSteps': numSteps, 'rewards': rewards}]
os.makedirs('output/epsilons', exist_ok=True)
file = 'output/epsilons/results.npy'
np.save(file, results)
print(results)


# %%
# decaying epsilon: linear decay
numSteps = np.zeros((10, 50))
rewards = np.zeros((10, 50))
results = {}
rats = 10
seed = SEED + 100
np.random.seed(seed)
for i in range(rats):
    print('----------------RAT: [' + str(i) + ']')
    QL = QLearn(i,
                outputDirectory='output/epsilons/linearDecay1-0/')
    QL.saveWeights('0')
    for j in range(50):
        [eligibilityTrace, hasGoal, currentStep,
            trialReward, ratCoords] = QL.runTrial(1-1/49*j)
        numSteps[i, j] = currentStep
        rewards[i, j] = trialReward
results = {'epsilon': 'linear_decay',
           'numSteps': numSteps, 'rewards': rewards, 'comments': 'linear decay from 1 to 0'}
os.makedirs('output/epsilons/linearDecay1-0', exist_ok=True)
file = 'output/epsilons/linearDecay1-0/results.npy'
np.save(file, results)
print(results)

# %%
# math.log(10-(10-1)*(j/49),10)
# for j in range(50):
#    print(1-1*j/49)
# import math
# aray = []
# for j in range(50):
#     value = math.exp(-j/math.exp(1))
#     print(value)
#     aray = [*aray, value]
# plt.plot(aray)
# plt.show()
# decaying epsilon: log decay
numSteps = np.zeros((10, 50))
rewards = np.zeros((10, 50))
results = {}
rats = 10
seed = SEED + 110
np.random.seed(seed)
for i in range(rats):
    print('----------------RAT: [' + str(i) + ']')
    QL = QLearn(i,
                outputDirectory='output/epsilons/logDecay/')
    QL.saveWeights('0')
    for j in range(50):
        [eligibilityTrace, hasGoal, currentStep,
            trialReward, ratCoords] = QL.runTrial(math.log(10-(10-1)*(j/49), 10))
        numSteps[i, j] = currentStep
        rewards[i, j] = trialReward
results = {'epsilon': 'log',
           'numSteps': numSteps, 'rewards': rewards, 'comments': 'log decay from 1 to 0'}
os.makedirs('output/epsilons/logDecay', exist_ok=True)
file = 'output/epsilons/logDecay/results.npy'
np.save(file, results)
print(results)

# %%
# decaying epsilon: exp decay
numSteps = np.zeros((10, 50))
rewards = np.zeros((10, 50))
results = {}
rats = 10
seed = SEED + 120
np.random.seed(seed)
for i in range(rats):
    print('----------------RAT: [' + str(i) + ']')
    QL = QLearn(i,
                outputDirectory='output/epsilons/expDecay/')
    QL.saveWeights('0')
    for j in range(50):
        [eligibilityTrace, hasGoal, currentStep,
            trialReward, ratCoords] = QL.runTrial(math.exp(-j/math.exp(1)))
        numSteps[i, j] = currentStep
        rewards[i, j] = trialReward
results = {'epsilon': 'exp',
           'numSteps': numSteps, 'rewards': rewards, 'comments': 'exp decay from 1 to 0'}
os.makedirs('output/epsilons/expDecay', exist_ok=True)
file = 'output/epsilons/expDecay/results.npy'
np.save(file, results)
print(results)

# %%
# zero weights init


def initZeroWeights(self):
    return np.zeros(np.random.rand(self.grid._xAxisSteps * self.grid._yAxisSteps, self.grid._numDirections))


numSteps = np.zeros((10, 50))
rewards = np.zeros((10, 50))
results = {}
rats = 10
seed = SEED + 130
np.random.seed(seed)
for i in range(rats):
    print('----------------RAT: [' + str(i) + ']')
    QL = QLearn(i,
                outputDirectory='output/initZeroWeights/')
    QL.initWeights = initZeroWeights
    QL.saveWeights('0')
    for j in range(50):
        [eligibilityTrace, hasGoal, currentStep,
         trialReward, ratCoords] = QL.runTrial(0.5)
        numSteps[i, j] = currentStep
        rewards[i, j] = trialReward
results = {'epsilon': 0.5,
           'numSteps': numSteps, 'rewards': rewards, 'comments': '0 init weights'}
os.makedirs('output/initZeroWeights', exist_ok=True)
file = 'output/initZeroWeights/results.npy'
np.save(file, results)
print(results)


# %%
# init weights rand between 0 and 1
def initZeroToOneWeights(self):
    return np.array(np.random.rand(self.grid._xAxisSteps * self.grid._yAxisSteps, self.grid._numDirections))


numSteps = np.zeros((10, 50))
rewards = np.zeros((10, 50))
results = {}
rats = 10
seed = SEED + 140
np.random.seed(seed)
for i in range(rats):
    print('----------------RAT: [' + str(i) + ']')
    QL = QLearn(i,
                outputDirectory='output/initZeroToOneWeights/')
    QL.initWeights = initZeroToOneWeights
    QL.saveWeights('0')
    for j in range(50):
        [eligibilityTrace, hasGoal, currentStep,
         trialReward, ratCoords] = QL.runTrial(0.5)
        numSteps[i, j] = currentStep
        rewards[i, j] = trialReward
results = {'epsilon': 0.5,
           'numSteps': numSteps, 'rewards': rewards, 'comments': '0 to 1 init weights'}
os.makedirs('output/initZeroToOneWeights', exist_ok=True)
file = 'output/initZeroToOneWeights/results.npy'
np.save(file, results)
print(results)

# %%
# off-policy (Q-learning), catch NaN exceptions and continue with next rat
epsilons = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]
results = []
seed = SEED + 150
print('[Off-Policy Tests]')
for epsilon in epsilons:
    seed += 1
    np.random.seed(seed)
    print('----------------EPSILON: [' + str(epsilon) + ']')
    numSteps = np.zeros((10, 50))
    rewards = np.zeros((10, 50))
    rats = 10
    for i in range(rats):
        print('----------------RAT: [' + str(i) + ']')
        QL = QLearn(i,
                    outputDirectory='output/offPolicy/epsilons/' +
                    str(epsilon)+'/',
                    onPolicy=False)
        QL.saveWeights('0')
        try:
            for j in range(50):
                [eligibilityTrace, hasGoal, currentStep,
                    trialReward, ratCoords] = QL.runTrial(epsilon)
                numSteps[i, j] = currentStep
                rewards[i, j] = trialReward
        except:
            print('----------------RAT: [' + str(i) +
                  '] skipping due to Exception.....')
    results = [*results, {'epsilon': epsilon,
                          'numSteps': numSteps, 'rewards': rewards}]
os.makedirs('output/offPolicy/epsilons', exist_ok=True)
file = 'output/offPolicy/epsilons/results.npy'
np.save(file, results)
print(results)


# %%
# without eligibility trace
epsilons = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]
results = []
seed = SEED + 200
print('[without eligibility trace]')


def withoutEligibilityTrace(self):
    def update(currentCoords, direction, eligibilityTrace):
        eligibilityTrace = np.zeros(
            (self.grid._xAxisSteps * self.grid._yAxisSteps, self.grid._numDirections))
        eligibilityTrace[:, direction] = self.calculatePlaceCellActivity(
            currentCoords)
        return eligibilityTrace
    return update


for epsilon in epsilons:
    seed += 1
    np.random.seed(seed)
    print('----------------EPSILON: [' + str(epsilon) + ']')
    numSteps = np.zeros((10, 50))
    rewards = np.zeros((10, 50))
    rats = 10
    for i in range(rats):
        print('----------------RAT: [' + str(i) + ']')
        QL = QLearn(i,
                    outputDirectory='output/noEligibilityTrace/epsilons/' +
                    str(epsilon)+'/')
        QL.updateEligibilityTrace = withoutEligibilityTrace(QL)
        QL.saveWeights('0')
        for j in range(50):
            [eligibilityTrace, hasGoal, currentStep,
                trialReward, ratCoords] = QL.runTrial(epsilon)
            numSteps[i, j] = currentStep
            rewards[i, j] = trialReward
    results = [*results, {'epsilon': epsilon,
                          'numSteps': numSteps, 'rewards': rewards}]
os.makedirs('output/noEligibilityTrace/epsilons', exist_ok=True)
file = 'output/noEligibilityTrace/epsilons/results.npy'
np.save(file, results)
print(results)

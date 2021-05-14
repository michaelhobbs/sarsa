import sarsa

rats = 1
for i in range(rats):
    print('----------------RAT: [' + str(i) + ']')
    sarsa.runExperiment(i)

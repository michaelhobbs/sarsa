# import sarsa
import ql

rats = 1
for i in range(rats):
    print('----------------RAT: [' + str(i) + ']')
    # sarsa.runExperiment(i)
    QL = ql.QLearn(i)
    for j in range(50):
        QL.runTrial(0.5 - j*0.5/50)

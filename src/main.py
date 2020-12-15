import sarsa

rats = 10
for i in range(rats):
	print ('----------------RAT: [' + str(i) + ']')
	sarsa.runExperiment(i)
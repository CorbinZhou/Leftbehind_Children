import numpy as np 
import pandas as pd 
import random
from sklearn.svm import SVR
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error

def mutate(_chromesome):
	if random.random() < mutationRate:
		_chromesome[random.randint(0,4)][random.randint(0,4)] = random.randint(0,109)
	return _chromesome

data2013 = pd.read_csv('expamles20_0.txt').ix[0:93,1:]
data2014 = pd.read_csv('data_2014.txt')

features2013 = data2013.drop('odds',axis=1)
features2014 = data2014.drop('odds',axis=1)

target2013 = data2013['odds']
target2014 = data2014['odds']

testtime = 10

#set parameters
ppl = 5
mutationRate = 0.3
chromesome = []
R2 = []
avrR2 = 0
generation = 0

#initialisation
for i in range(ppl):
	chromesome.append([random.randint(0,109),random.randint(0,109),random.randint(0,109),random.randint(0,109),random.randint(0,109)])
	cv_avr = []
	for j in range(5):
		X_train, X_test, y_train, y_test = cross_validation.train_test_split(
	    	features2013[chromesome[i]], target2013, test_size=0.15, random_state=j)
		#evaluate fitness function
		SVRmodel = SVR(C=1e2, gamma=0.05, kernel='rbf')
		SVRmodel.fit(X_train,y_train)
		cv_avr.append(SVRmodel.score(X_test,y_test))
	R2.append(1.0*sum(cv_avr)/len(cv_avr))
avrR2 = 1.0*(sum(R2)/len(R2))

f = open('GAdata'+str(testtime)+'.txt','wb')
f.write('generation,train,validation,test\n')
f2 = open('features'+str(testtime)+'.txt','wb')

#training & GA
while (avrR2 < 0.9 and generation < 300):
	#chromesome selection
	sumR2 = [R2[0]]
	for i in range(len(R2)-1):
		sumR2.append(R2[i+1]+R2[i])
	next = [0,1,2,3,4]
	parents = []
	drop = []
	r = random.uniform(0,sumR2[len(R2)-1])
	if r < sumR2[0]:
		parents.append(chromesome[0])
		del chromesome[0]
		drop.append(0)
	elif r < sumR2[1]:
		parents.append(chromesome[1])
		del chromesome[1]
		drop.append(1)
	elif r < sumR2[2]:
		parents.append(chromesome[2])
		del chromesome[2]
		drop.append(2)
	elif r < sumR2[3]:
		parents.append(chromesome[3])
		del chromesome[3]
		drop.append(3)
	else:
		parents.append(chromesome[4])
		del chromesome[4]
		drop.append(4)

	del R2[drop[0]]
	del next[drop[0]]

	sumR2 = [R2[0]]
	for i in range(len(R2)-1):
		sumR2.append(R2[i+1]+R2[i])
	r = random.uniform(0,sumR2[len(R2)-1])
	if r < sumR2[0]:
		parents.append(chromesome[0])
		del chromesome[0]
		drop.append(0)
	elif r < sumR2[1]:
		parents.append(chromesome[1])
		del chromesome[1]
		drop.append(1)
	elif r < sumR2[2]:
		parents.append(chromesome[2])
		del chromesome[2]
		drop.append(2)
	else:
		parents.append(chromesome[3])
		del chromesome[3]
		drop.append(3)

	del R2[drop[1]]
	del next[drop[1]]

	sumR2 = [R2[0]]
	for i in range(len(R2)-1):
		sumR2.append(R2[i+1]+R2[i])
	r = random.uniform(0,sumR2[len(R2)-1])
	if r < sumR2[0]:
		parents.append(chromesome[0])
	elif r < sumR2[1]:
		parents.append(chromesome[1])
	else:
		parents.append(chromesome[2])

	#crossover
	offsprings = [	parents[0][0:3]+parents[1][3:5],
					parents[1][0:3]+parents[0][3:5],
					parents[0][0:3]+parents[2][3:5],
					parents[2][0:2]+parents[0][2:5],
					parents[1][0:2]+parents[2][2:5]
					]
	#mutation
	offsprings = mutate(offsprings)
	# offsprings = mutate(offsprings)
	# offsprings = mutate(offsprings)

	#performance validation
	chromesome = offsprings
	R2 = []
	tr_R2 = []
	for i in range(len(chromesome)):
		#cross-validation
		mse_avr = []
		tr_avr = []
		cv_avr = []
		mse = []
		for j in range(5):
			X_train, X_test, y_train, y_test = cross_validation.train_test_split(
		    	features2013[chromesome[i]], target2013, test_size=0.15, random_state=j)
			#evaluate fitness function
			SVRmodel = SVR(C=1e2, gamma=0.05, kernel='rbf')
			SVRmodel.fit(X_train,y_train)
			tr_avr.append(SVRmodel.score(X_train,y_train))
			cv_avr.append(SVRmodel.score(X_test,y_test))
			mse_avr.append(mean_squared_error(SVRmodel.predict(X_test),y_test))
		tr_R2.append(1.0*sum(tr_avr)/len(tr_avr))
		R2.append(1.0*sum(cv_avr)/len(cv_avr))
		mse.append(1.0*sum(mse_avr)/len(mse_avr))
	avrMse = 1.0*sum(mse)/len(mse)
	avrR2 = 1.0*(sum(R2)/len(R2))
	avrTrR2 = 1.0*(sum(tr_R2)/len(tr_R2))

	#test performance
	testR2 = []
	testMse = []
	for i in range(len(chromesome)):
		SVRmodel = SVR(C=1e2, gamma=0.05, kernel='rbf')
		SVRmodel.fit(data2013[chromesome[i]],target2013)
		testR2.append(SVRmodel.score(data2014[chromesome[i]],target2014))
		testMse.append(mean_squared_error(SVRmodel.predict(data2014[chromesome[i]]),target2014))
	test = 1.0*sum(testR2)/len(testR2)
	testAvrMse = 1.0*sum(testMse)/len(testMse)
	
	print generation
	print avrTrR2
	print avrR2
	print test

	f.write(str(generation)+','+str(avrTrR2)+','+str(avrR2)+','+str(test)+','+str(avrMse)+','+str(testAvrMse)+'\n')
	f2.write('generation: '+str(generation)+'\n')
	f2.write('features: \n')
	for ii in range(len(chromesome)):
		for jj in range(len(chromesome[ii])):
			f2.write('\t'+str(features2013[chromesome[ii]].columns.values[jj])+'\n')
		f2.write('\n')
	f2.write('trainR2:\n')
	f2.write(str(tr_R2)+'\n')
	f2.write('validR2:\n')
	f2.write(str(R2)+'\n')
	f2.write('testR2:\n')
	f2.write(str(testR2)+'\n')
	f2.write('validmse:\n')
	f2.write(str(mse)+'\n')
	f2.write('testmse:\n')
	f2.write(str(testMse)+'\n')

	generation += 1

f.close()
f2.close()













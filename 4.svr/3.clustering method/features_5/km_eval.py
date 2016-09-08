import numpy as np
import pandas as pd
from sklearn.svm import SVR
import random
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error

data2013 = pd.read_csv('expamles20_0.txt').ix[0:93]
data2014 = pd.read_csv('data_2014.txt')

features2013 = data2013.drop('odds',axis=1)
features2014 = data2014.drop('odds',axis=1)

target2013 = data2013['odds']
target2014 = data2014['odds']

class_ = []
for i in range(5):
	class_.append(pd.read_csv('class_'+str(i)+'.txt')['name'].values)

svr_model = SVR(kernel='rbf', C=1e2, gamma=0.05)

i = random.randint(0,len(class_[0])-1)
j = random.randint(0,len(class_[1])-1)
k = random.randint(0,len(class_[2])-1)
l = random.randint(0,len(class_[3])-1)
m = random.randint(0,len(class_[4])-1)

cv_avr = []
for fold in range(10):
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    	features2013[[class_[0][i],class_[1][j],class_[2][k],class_[3][l],class_[4][m]]],target2013, test_size=0.15, random_state=fold)
	#evaluate fitness function
	SVRmodel = SVR(C=1e2, epsilon=0.05, kernel='rbf')
	SVRmodel.fit(X_train,y_train)
	cv_avr.append(SVRmodel.score(X_test,y_test))
R2 = 1.0*sum(cv_avr)/len(cv_avr)

f = open('KMdata10.txt','wb')
f.write('generation,train,validation,test,validmse,testmse\n')
f2 = open('features10.txt','wb')
iteration = 0

while (R2 < 0.85 and iteration < 500):
	old = i,j,k,l,m
	r = random.random()
	if r < 0.2:
		i = random.randint(0,len(class_[0])-1)
	elif r < 0.4:
		j = random.randint(0,len(class_[1])-1)
	elif r < 0.6:
		k = random.randint(0,len(class_[2])-1)
	elif r < 0.8:
		l = random.randint(0,len(class_[3])-1)
	else:
		m = random.randint(0,len(class_[4])-1)

	rr = random.random()
	if rr < 0.8:
		r = random.random()
		if r < 0.2:
			i = random.randint(0,len(class_[0])-1)
		elif r < 0.4:
			j = random.randint(0,len(class_[1])-1)
		elif r < 0.6:
			k = random.randint(0,len(class_[2])-1)
		elif r < 0.8:
			l = random.randint(0,len(class_[3])-1)
		else:
			m = random.randint(0,len(class_[4])-1)

	cv_avr = []
	mse_avr = []
	tr_avr = []
	for fold in range(5):
		X_train, X_test, y_train, y_test = cross_validation.train_test_split(
	    	features2013[[class_[0][i],class_[1][j],class_[2][k],class_[3][l],class_[4][m]]],target2013, test_size=0.15, random_state=fold)
		#evaluate fitness function
		SVRmodel = SVR(C=1e2,gamma=0.05, kernel='rbf')
		SVRmodel.fit(X_train,y_train)
		tr_avr.append(SVRmodel.score(X_train,y_train))
		cv_avr.append(SVRmodel.score(X_test,y_test))
		mse_avr.append(mean_squared_error(SVRmodel.predict(X_test),y_test))
	tr_R2 = 1.0*sum(tr_avr)/len(tr_avr)
	newR2 = 1.0*sum(cv_avr)/len(cv_avr)
	mse = 1.0*sum(mse_avr)/len(mse_avr)

	if newR2 <= R2:
		i,j,k,l,m = old
	
	print iteration
	print i,j,k,l,m
	# print features2013[[class_[0][i],class_[1][j],class_[2][k],class_[3][l],class_[4][m]]].columns
	print newR2

	SVRmodel = SVR(C=1e2, gamma=0.05, kernel='rbf')
	SVRmodel.fit(features2013[[class_[0][i],class_[1][j],class_[2][k],class_[3][l],class_[4][m]]],target2013)
	test_R2 = SVRmodel.score(features2014[[class_[0][i],class_[1][j],class_[2][k],class_[3][l],class_[4][m]]],target2014)
	test_mse = mean_squared_error(SVRmodel.predict(features2014[[class_[0][i],class_[1][j],class_[2][k],class_[3][l],class_[4][m]]]),target2014)
	
	f.write(str(iteration)+','+str(tr_R2)+','+str(newR2)+','+str(test_R2)+','+str(mse)+','+str(test_mse)+'\n')
	f2.write('iteration: '+str(iteration)+'\n')
	f2.write('features: '+'\n')
	for ii in range(5):
		f2.write('\t'+str(features2013[[class_[0][i],class_[1][j],
					class_[2][k],class_[3][l],
					class_[4][m]]].columns.values[ii])+'\n')
	f2.write('\n')
	R2 = newR2

	iteration += 1

f.close()
f2.close()

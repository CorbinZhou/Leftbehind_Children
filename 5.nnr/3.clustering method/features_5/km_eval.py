import numpy as np
import pandas as pd
from sklearn.svm import SVR
import random
from sklearn import cross_validation
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.regularizers import l2
from sklearn.metrics import r2_score
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

model = Sequential()
model.add(Dense(5, input_dim=5,W_regularizer=l2(0.01)))
model.add(Activation('relu'))
model.add(Dense(1,W_regularizer=l2(0.01)))
model.add(Activation('tanh'))

i = random.randint(0,len(class_[0])-1)
j = random.randint(0,len(class_[1])-1)
k = random.randint(0,len(class_[2])-1)
l = random.randint(0,len(class_[3])-1)
m = random.randint(0,len(class_[4])-1)

tr_R2 = []
r2 = []
mse = []
for fold in range(5):
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    	features2013[[class_[0][i],class_[1][j],class_[2][k],class_[3][l],class_[4][m]]],target2013, test_size=0.15, random_state=fold)
	#evaluate fitness function
	model.compile(loss='mean_squared_error', optimizer='sgd')
	model.fit(X_train.values, y_train.values, nb_epoch=5, batch_size=10)
	pre_ = model.predict_proba(X_test.values,batch_size=10)
	trained_ = model.predict(X_train.values,batch_size=10)

	tr_R2.append(r2_score(y_train.values, trained_))
	r2.append(r2_score(y_test.values, pre_))
	mse.append(mean_squared_error(y_test.values, pre_))
avrMse = 1.0*sum(mse)/len(mse)
avrR2 = 1.0*(sum(r2)/len(r2))
avrTrR2 = 1.0*(sum(tr_R2)/len(tr_R2))

f = open('KMdata5.txt','wb')
f.write('generation,train,validation,test\n')
f2 = open('features5.txt','wb')
iteration = 0

while (avrR2 < 0.85 and iteration < 30):
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

	r2 = []
	tr_R2 = []
	mse = []
	for fold in range(10):
		X_train, X_test, y_train, y_test = cross_validation.train_test_split(
	    	features2013[[class_[0][i],class_[1][j],class_[2][k],class_[3][l],class_[4][m]]],target2013, test_size=0.15, random_state=fold)
		#evaluate fitness function
		model.compile(loss='mean_squared_error', optimizer='sgd')
		model.fit(X_train.values, y_train.values, nb_epoch=5, batch_size=10)
		trained_ = model.predict(X_train.values,batch_size=10)
		pre_ = model.predict(X_test.values,batch_size=10)


		tr_R2.append(r2_score(y_train.values, trained_))
		r2.append(r2_score(y_test.values, pre_))
		mse.append(mean_squared_error(y_test.values, pre_))
	avrMse = 1.0*sum(mse)/len(mse)
	newavrR2 = 1.0*(sum(r2)/len(r2))
	avrTrR2 = 1.0*(sum(tr_R2)/len(tr_R2))

	if newavrR2 <= avrR2:
		i,j,k,l,m = old
	
	print iteration
	print i,j,k,l,m
	# print features2013[[class_[0][i],class_[1][j],class_[2][k],class_[3][l],class_[4][m]]].columns
	print newavrR2

	model.compile(loss='mean_squared_error', optimizer='sgd')
	model.fit(features2013[[class_[0][i],class_[1][j],class_[2][k],class_[3][l],class_[4][m]]].values,target2013.values, nb_epoch=5, batch_size=5)
	pre_ = model.predict_proba(features2014[[class_[0][i],class_[1][j],class_[2][k],class_[3][l],class_[4][m]]].values,batch_size=5)
	testR2 = r2_score(target2014.values, pre_)
	testMse = mean_squared_error(target2014.values, pre_)

	f.write(str(iteration)+','+str(avrTrR2)+','+str(avrR2)+','+str(testR2)+','+str(avrMse)+','+str(testMse)+'\n')
	f2.write('iteration: '+str(iteration)+'\n')
	f2.write('features: '+'\n')
	for ii in range(5):
		f2.write('\t'+str(features2013[[class_[0][i],class_[1][j],
					class_[2][k],class_[3][l],
					class_[4][m]]].columns.values[ii])+'\n')
	f2.write('\n')
	f2.write('trainR2:\n')
	f2.write(str(tr_R2)+'\n')
	f2.write('validR2:\n')
	f2.write(str(r2)+'\n')
	f2.write('testR2:\n')
	f2.write(str(testR2)+'\n')
	f2.write('validmse:\n')
	f2.write(str(mse)+'\n')
	f2.write('testmse:\n')
	f2.write(str(testMse)+'\n')
	avrR2 = newavrR2

	iteration += 1

f.close()
f2.close()

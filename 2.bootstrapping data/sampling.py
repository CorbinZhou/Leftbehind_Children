import numpy as np
import pandas as pd

def gNoise():
	noise = pd.DataFrame(np.random.normal(0,0.01,size = (oldSamples.index.shape[0],oldSamples.columns.shape[0])))
	noise.index = oldSamples.index
	noise.columns = oldSamples.columns
	return noise

oldSamples = pd.read_csv('data_2013.txt')

for j in range(20):
	new = oldSamples.add(gNoise())
	for i in range(19):
		temp = oldSamples.add(gNoise())
		new = pd.concat([new,temp])

	# new = new.drop(['Unnamed: 0', 'Unnamed: 0.1'],axis=1)
	# print new.columns
	new.to_csv('expamles20_'+str(j)+'.txt',index=False)

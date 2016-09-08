import numpy as np 
import pandas as pd
import re

filenum = 1
features = []

for num in range(10):
	filenum = num + 1
	validation = pd.read_csv('KMdata'+str(filenum)+'.txt')
	with open('features'+str(filenum)+'.txt') as f:
	    content = f.readlines()

	for i in range(500):
		k = i*8+2
		new = []
		new.append(re.sub('\s+', '',content[k]))
		new.append(re.sub('\s+', '',content[k+1]))
		new.append(re.sub('\s+', '',content[k+2]))
		new.append(re.sub('\s+', '',content[k+3]))
		new.append(re.sub('\s+', '',content[k+4]))
		new.append(validation.index[i]+num*300)
		new.append(validation['validation'][i])
		features.append(new)

df_features = pd.DataFrame.from_records(features)
df_features.columns = ['f1','f2','f3','f4','f5','num','validation']
sort_df = df_features.sort(['validation'],ascending=False)
sort_df.to_csv('sorted_features.txt',index = False)
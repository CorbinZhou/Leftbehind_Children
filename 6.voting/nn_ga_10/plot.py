import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import random

data = pd.read_csv('voting_result.txt')
legend = pd.read_csv('class/field.txt')

for i in range(28):
	if data[str(i+1)][len(data[str(i+1)])-1] != 0:
		print legend['Name'].values[i]
		random.seed(i+1)
		plt.plot(data['n'],data[str(i+1)],label = legend['Name'].values[i],c =(random.random(),random.random(),random.random()))
	# plt.hold(True)

plt.legend(loc='upper left',
			fontsize = 10,
			bbox_to_anchor=[0.0, 1.0], 
           	columnspacing=0.5, labelspacing=0.0,
           	handletextpad=0.0, handlelength=1.5,
           	fancybox=True)

plt.savefig('voting.jpg')
plt.show()

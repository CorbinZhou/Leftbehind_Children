import numpy as np 
import pandas as pd
import re

superf = {}
for i in range(25):
	with open('class/field'+str(i+1)+'.txt') as f:
		content = f.readlines()
		for j in range(len(content)):
			content[j] = re.sub('\s+', '',content[j])
	superf[i] = content

data = pd.read_csv('sorted_features.txt')

result = {
	'n':[0],1:[0], 2:[0], 3:[0],4:[0],5:[0],6:[0],7:[0],8:[0],9:[0],10:[0],
	11:[0],12:[0],13:[0],14:[0],15:[0],16:[0],17:[0],18:[0],19:[0],20:[0],
	21:[0],22:[0],23:[0],24:[0],25:[0],26:[0],27:[0],28:[0]
}
res_df = pd.DataFrame.from_dict(result,orient='columns')

r = 0
for index, row in data.iterrows():
	if r < 180 and r > 30:
		df = pd.DataFrame(res_df.tail(n=1))
		df['n'] = df['n']+1
		for i in range(25):
			if row['f1'] in superf[i]:
				df[i+1] = df[i+1]+1
			if row['f2'] in superf[i]:
				df[i+1] = df[i+1]+1
			if row['f3'] in superf[i]:
				df[i+1] = df[i+1]+1
			if row['f4'] in superf[i]:
				df[i+1] = df[i+1]+1
			if row['f5'] in superf[i]:
				df[i+1] = df[i+1]+1
			if row['f6'] in superf[i]:
				df[i+1] = df[i+1]+1
			if row['f7'] in superf[i]:
				df[i+1] = df[i+1]+1
			if row['f8'] in superf[i]:
				df[i+1] = df[i+1]+1
			if row['f9'] in superf[i]:
				df[i+1] = df[i+1]+1
			if row['f10'] in superf[i]:
				df[i+1] = df[i+1]+1

		res_df = pd.concat([res_df, df],ignore_index=True)
		print df
	r += 1

res_df.to_csv('voting_result.txt',index = False)


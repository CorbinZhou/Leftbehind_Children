import numpy as np
import pandas as pd
from minepy import MINE

data = pd.read_csv('data_2013.txt')

mine = MINE(alpha=0.6, c=15)

result = {}
for column in data:
	mine.compute_score(data[column], data['odds'])
	result[column] = mine.mic()

df = pd.DataFrame.from_dict(result,orient='index')
df.columns = ['mic']
sorted_ = df.sort('mic',ascending = 0)

# print sorted_
sorted_.to_csv('mic_result.txt')

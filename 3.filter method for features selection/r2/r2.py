import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

data = pd.read_csv('data_2013.txt')

result = {}
for column in data:
	result[column] = r2_score(data['odds'], data[column])

df = pd.DataFrame.from_dict(result,orient='index')
df.columns = ['r2']
sorted_ = df.sort('r2',ascending = 0)

# print sorted_
sorted_.to_csv('r2_result.txt')

import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr

data = pd.read_csv('data_2013.txt')

result = {}
for column in data:
	cc = pearsonr(data[column],data['odds'])[0]
	result[column] = cc

df = pd.DataFrame.from_dict(result,orient='index')
df.columns = ['cc']
sorted_ = df.sort('cc',ascending = 0)

sorted_.to_csv('cc_result.txt')

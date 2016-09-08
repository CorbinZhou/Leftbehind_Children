import numpy as np 
import pandas as pd 

data2013 = pd.read_csv('data_2013.txt')
data2014 = pd.read_csv('data_2014.txt')

columns2013 = set(data2013.columns.values)
columns2014 = set(data2014.columns.values)

print len(columns2013)
print len(columns2014)
print len(columns2013 & columns2014)
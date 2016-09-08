# coding: utf-8
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

data2013 = pd.read_csv('expamles20_0.txt').ix[0:93]
data2014 = pd.read_csv('data_2014.txt')

selected_features = pd.read_csv('selected_features.txt')

svr_model = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_model.fit(data2013[selected_features['name']],data2013['odds'])
r2_5_features = svr_model.score(data2014[selected_features['name']],data2014['odds'])
mse_5_features = mean_squared_error(svr_model.predict(data2014[selected_features.name[0:5]]),data2014['odds'])

f5 = open('svr.txt','wb')

pre = svr_model.predict(data2014[selected_features['name']])
f5.write('predict: ')
f5.write(str(pre))
print pre
f5.write('\n')
plt.plot(range(31),pre,label = 'predict*1.0')

# data2014[selected_features['name'][2]] = data2014[selected_features['name'][2]]*1.1
data2014[selected_features['name'][3]] = data2014[selected_features['name'][3]]*1.1
data2014[selected_features['name'][4]] = data2014[selected_features['name'][4]]*1.1

pre_1_1 = svr_model.predict(data2014[selected_features['name']])
f5.write('predict_1.1: ')
f5.write(str(pre_1_1))
print pre_1_1
f5.write('\n')
plt.plot(range(31),pre_1_1,label = str(1.1))

# data2014[selected_features['name'][2]] = data2014[selected_features['name'][2]]/1.1*1.2
data2014[selected_features['name'][3]] = data2014[selected_features['name'][3]]/1.1*1.2
data2014[selected_features['name'][4]] = data2014[selected_features['name'][4]]/1.1*1.2

pre_1_2 = svr_model.predict(data2014[selected_features['name']])
f5.write('predict_1.2: ')
f5.write(str(pre_1_2))
print pre_1_2
f5.write('\n')
plt.plot(range(31),pre_1_2,label = str(1.2))

# data2014[selected_features['name'][2]] = data2014[selected_features['name'][2]]/1.2*0.8
data2014[selected_features['name'][3]] = data2014[selected_features['name'][3]]/1.2*0.8
data2014[selected_features['name'][4]] = data2014[selected_features['name'][4]]/1.2*0.8

pre_0_8 = svr_model.predict(data2014[selected_features['name']])
f5.write('predict_0.8: ')
f5.write(str(pre_0_8))
print pre_0_8
f5.write('\n')
plt.plot(range(31),pre_0_8,label = str(0.8))

plt.xlabel('province')
plt.legend()
plt.savefig('plot.jpg')
plt.show()

f5.write('\n')
f5.write('true values: ')
f5.write(str(data2014['odds'].values))
print data2014['odds'].values
f5.write('\n')

f5.write('r2: ')
f5.write(str(r2_5_features))
f5.write('\n')

f5.write('r2: ')
f5.write(str(r2_5_features))
f5.write('\n')
f5.write('mse: ')
f5.write(str(mse_5_features))
f5.write('\n')
f5.write('SV: \n')
f5.write(str(svr_model.support_vectors_))
f5.write('\n')
f5.write('dual_coef: \n')
f5.write(str(svr_model.dual_coef_))
f5.write('\n')
f5.write('intercept: \n')
f5.write(str(svr_model.intercept_))
f5.write('\n')
f5.close()

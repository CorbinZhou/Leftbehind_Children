import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

data2013 = pd.read_csv('data_2013.txt')
data2014 = pd.read_csv('data_2014.txt')

# data = pd.concat([data2013,data2014],ignore_index=True)

selected_features = pd.read_csv('selected_features_15_mic_2013.txt')

svr_model = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_model.fit(data2013[selected_features.name[0:5]],data2013['odds'])
r2_5_features = svr_model.score(data2014[selected_features.name[0:5]],data2014['odds'])
mse_5_features = mean_squared_error(svr_model.predict(data2014[selected_features.name[0:5]]),data2014['odds'])

f5 = open('svr_mic_result_5.txt','wb')
f5.write('r2_5_features: ')
f5.write(str(r2_5_features))
f5.write('\n')
f5.write('mse_5_features: ')
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

svr_model = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_model.fit(data2013[selected_features.name[0:10]],data2013['odds'])
r2_10_features = svr_model.score(data2014[selected_features.name[0:10]],data2014['odds'])
mse_10_features = mean_squared_error(svr_model.predict(data2014[selected_features.name[0:10]]),data2014['odds'])

f10 = open('svr_mic_result_10.txt','wb')
f10.write('r2_10_features: ')
f10.write(str(r2_10_features))
f10.write('\n')
f10.write('mse_10_features: ')
f10.write(str(mse_10_features))
f10.write('\n')
f10.write('SV: \n')
f10.write(str(svr_model.support_vectors_))
f10.write('\n')
f10.write('dual_coef: \n')
f10.write(str(svr_model.dual_coef_))
f10.write('\n')
f10.write('intercept: \n')
f10.write(str(svr_model.intercept_))
f10.write('\n')
f10.close()


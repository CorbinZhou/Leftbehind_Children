import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

data2013 = pd.read_csv('data_2013.txt')
data2014 = pd.read_csv('data_2014.txt')

# data = pd.concat([data2013,data2014],ignore_index=True)

selected_features = pd.read_csv('selected_feature_10.txt')

svr_model = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_model.fit(data2013[selected_features.name[0:10]],data2013['odds'])
r2_10_features = svr_model.score(data2014[selected_features.name[0:10]],data2014['odds'])
mse_10_features = mean_squared_error(svr_model.predict(data2014[selected_features.name[0:10]]),data2014['odds'])

f5 = open('svr_cc_result_10.txt','wb')
f5.write('r2_10_features: ')
f5.write(str(r2_10_features))
f5.write('\n')
f5.write('mse_5_features: ')
f5.write(str(mse_10_features))
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
f5.write('predict:')
f5.write(str(svr_model.predict(data2014[selected_features.name[0:10]])))
f5.write('\n')
f5.close()


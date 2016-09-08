import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from sklearn.metrics import r2_score
from keras.regularizers import l2
from sklearn.metrics import mean_squared_error

data2013 = pd.read_csv('expamles20_0.txt').ix[0:93]
data2014 = pd.read_csv('data_2014.txt')

selected_features = pd.read_csv('selected_features_15_cc_2013.txt')


model = Sequential()
model.add(Dense(5, input_dim=5,W_regularizer=l2(0.01)))
model.add(Activation('relu'))
model.add(Dense(1,W_regularizer=l2(0.01)))
model.add(Activation('tanh'))

model.compile(loss='mean_absolute_error', optimizer='sgd')
model.fit(data2013[selected_features.name[0:5]].values,data2013['odds'].values,nb_epoch=20, batch_size=5)
pre_ = model.predict(data2014[selected_features.name[0:5]].values,batch_size=5)
r2_5_features = r2_score(data2014['odds'].values, pre_)
mse_5_features = mean_squared_error(data2014['odds'].values, pre_)

f5 = open('nn_cc_result_5.txt','ab')
f5.write(str(r2_5_features)+','+str(mse_5_features))
f5.write('\n')
f5.close()

model = Sequential()
model.add(Dense(8, input_dim=10,W_regularizer=l2(0.01)))
model.add(Activation('relu'))
model.add(Dense(1,W_regularizer=l2(0.01)))
model.add(Activation('tanh'))

model.compile(loss='mean_absolute_error', optimizer='sgd')
model.fit(data2013[selected_features.name[0:10]].values,data2013['odds'].values,nb_epoch=20, batch_size=5)
pre_ = model.predict(data2014[selected_features.name[0:10]].values,batch_size=5)
r2_10_features = r2_score(data2014['odds'].values, pre_)
mse_10_features = mean_squared_error(data2014['odds'].values, pre_)

f10 = open('nn_cc_result_10.txt','ab')
f10.write(str(r2_10_features)+','+str(mse_10_features))
f10.write('\n')

f10.close()
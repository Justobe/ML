# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 14:08:39 2019

@author: yanming
"""
# In[1] 
from argparse import Namespace
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model.stochastic_gradient import SGDRegressor
import utils as uls

args = Namespace(
    data_num=100,
    train_rate=0.8,
    test_rate=0.2,
    epoches=150
)


# In[2]
x, y, k = uls.get_linear_data(args.data_num)
data = np.vstack([x,y]).T  # Stack two 1-d arrays vertically

df = pd.DataFrame(data,columns=['X','Y'])
plt.title("Random data for Linear Regression")
plt.scatter(df['X'], df['Y'])
plt.show()

# In[3] standardized and normalized
X_train, X_test, y_train, y_test = train_test_split(df['X'], df['Y'], test_size=args.test_rate)

x_scaler = StandardScaler().fit(X_train.values.reshape(-1, 1))  # reshape is deprecated.Please use .values.reshape
y_scaler = StandardScaler().fit(y_train.values.reshape(-1, 1))

standard_x_train = x_scaler.transform(X_train.values.reshape(-1, 1))
standard_x_test = x_scaler.transform(X_test.values.reshape(-1, 1))
standard_y_train = y_scaler.transform(y_train.values.reshape(-1, 1))
standard_y_test = y_scaler.transform(y_test.values.reshape(-1, 1))

# In[4] linear regression
linear_model = SGDRegressor(loss='squared_loss', max_iter=args.epoches,penalty='none')

# Regularization
# linear_model = SGDRegressor(loss='squared_loss', max_iter=args.epoches,penalty='l2')
linear_model.fit(standard_x_train, standard_y_train)


# In[5] get prediction data
prediction_train = (linear_model.predict(standard_x_train)*np.sqrt(y_scaler.var_) + y_scaler.mean_)
prediction_test = (linear_model.predict(standard_x_test)*np.sqrt(y_scaler.var_) + y_scaler.mean_)

# In[6] Evaluation

train_MSE = np.mean((y_train - prediction_train)**2)
test_MSE = np.mean((y_test - prediction_test)**2)

predict_k = linear_model.coef_*(y_scaler.scale_/x_scaler.scale_)

# return coefficients is 1d array-like
print("Really coefficients is {} and predict coefficients is {}.".format(k,predict_k[0]))

plt.figure(figsize=(15,5))
plt.subplot(1, 2, 1)
plt.title("Train")
plt.scatter(X_train,y_train,label='y_train')
plt.plot(X_train,prediction_train,label='lm',color='red',linewidth=1,linestyle='-')

plt.subplot(1, 2, 2)
plt.title("Test")
plt.scatter(X_test,y_test,label='y_test')
plt.plot(X_test,prediction_test,label='lm',color='red',linewidth=1,linestyle='-')

plt.show()

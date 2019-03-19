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

args = Namespace(
    data_num=100,
    train_rate=0.8,
    test_rate=0.2,
    epoches=150
)


def get_linear_data(num):
    if num <= 1:
        print("The length of data should be more than 1!")
        pass
    else:   
        x = np.arange(1,num+1) # get axis x     
        rand_k = 5 + (5-3)*np.random.random() # get random k in range [1,3]
        rand_noise = np.random.uniform(-1,1,num)*(num/10) # get random noise
        y = rand_k*x + rand_noise #
        return x, y


# In[2]
x, y = get_linear_data(args.data_num)
data = np.vstack([x,y]).T  # 将两个一位数组沿着垂直方向堆叠起来

df = pd.DataFrame(data,columns=['X','Y'])
plt.title("Random data for Linear Regression")
plt.scatter(df['X'], df['Y'])
plt.show()

# In[3] 对数据进行标准化，归一化
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

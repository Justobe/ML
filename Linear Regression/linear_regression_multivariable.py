# In[1]
import numpy as np
import pandas as pd
from argparse import Namespace
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.linear_model.stochastic_gradient import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression  # Compute the ANOVA F-value for the provided sample
args = Namespace(
    train_rate=0.8,
    test_rate=0.2,
    epoches=1000
)

# In[2] load dataset
boston = load_boston(return_X_y=False)
boston_X = pd.DataFrame(boston.data,columns=boston.feature_names)
boston_y = boston.target

raw_X_train,raw_X_test,y_train,y_test = train_test_split(boston_X,boston_y,test_size=args.test_rate)
scores = []
for k_dim in range(1, 14): # select dimension from data
    selector = SelectKBest(f_regression, k=k_dim)
    bestFeature = selector.fit(boston_X,boston_y)
    cols = boston.feature_names[selector.get_support()]
    X_train = raw_X_train[cols]
    X_test = raw_X_test[cols]
    # In[3] standardized and normalized
    x_scale = StandardScaler().fit(X_train)
    y_scale = StandardScaler().fit(y_train.reshape(-1,1))

    standard_X_train = x_scale.transform(X_train)
    standard_X_test = x_scale.transform(X_test)
    standard_y_train = y_scale.transform(y_train.reshape(-1,1))
    standard_y_test = y_scale.transform(y_test.reshape(-1,1))
    # In[4] train regularization
    total_score = 0
    # from every dimension selection; compute for 20 times
    for i in range(20):
        linear_model = SGDRegressor(loss='squared_loss', penalty='l2',max_iter=args.epoches)
        linear_model.fit(standard_X_train,standard_y_train)

        prediction_train = linear_model.predict(standard_X_train)*np.sqrt(y_scale.var_) + y_scale.mean_
        prediction_test = linear_model.predict(standard_X_test)*np.sqrt(y_scale.var_) + y_scale.mean_
        # In[5] Evaluation
        train_mse = np.mean((prediction_train - y_train)**2)
        test_mse = np.mean((prediction_test - y_test)**2)
        total_score += linear_model.score(standard_X_test, standard_y_test)
    scores.append(total_score/20.0)
    print("Score of model when {} dim are selected: {}".format(k_dim,total_score/20.0))  # avg score

plt.figure(figsize=(10,10))
plt.plot(scores,'ro-',label="score")
plt.legend()
plt.show()
plt.savefig("scores.png")


plt.figure(figsize=(10,10))
plt.plot(prediction_train[:50],'ro-',label="predict")
plt.plot(y_train[:50],'go--',label="groundtruth")
plt.title("Linear Regression")
plt.xlabel("sample index")
plt.ylabel("house price")
plt.legend()
plt.show()
plt.savefig("predict.png")
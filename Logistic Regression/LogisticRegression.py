# In[1]
from urllib import request
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from argparse import Namespace

args = Namespace(
    train_rate=0.8,
    test_rate=0.2,
    epoches=500,
)

# In[2]
def get_csv(u, file):
    if not os.path.exists(file):
        response = request.urlopen(u)
        content = response.read()
        with open(file, "wb") as f:
            f.write(content)


url = "https://raw.githubusercontent.com/GokuMohandas/practicalAI/master/data/titanic.csv"
filename = "titanic.csv"
get_csv(url,filename)

# In[3] get data from csv
data = pd.read_csv(filename)
df = pd.DataFrame(data)
print(df.isnull().sum())

categorical_feature = ['sex','embarked','pclass']  # dummy category label
drop_columns = ['name', 'cabin', 'ticket']  # drop nosense columns
df.drop(drop_columns,axis=1,inplace=True)
df = pd.get_dummies(df,columns=categorical_feature)
df.fillna(df.mean()['age':'fare'],inplace=True)  # fill na
print(df.isnull().sum())

# In[4] get data and target;split the dataset
data_y = df['survived']
data_X = df.drop(['survived'],axis=1)

X_train,X_test,y_train,y_test = train_test_split(data_X,data_y,test_size=args.test_rate)

# In[5] Standardize and normalization for x
x_scale = StandardScaler().fit(X_train.values)
y_scale = StandardScaler().fit(y_train.values.reshape(-1,1))

standard_X_train = x_scale.transform(X_train)
standard_X_test = x_scale.transform(X_test)

# In[6] train model
log_reg = SGDClassifier(loss='log',penalty='none',max_iter=args.epoches)
log_reg.fit(X=standard_X_train,y=y_train)

# In[7] Evaluation
pred_train = log_reg.predict(standard_X_train)
pred_test = log_reg.predict(standard_X_test)
test_score = accuracy_score(y_test,log_reg.predict(standard_X_test))

# Accuracy
train_acc = accuracy_score(y_train, pred_train)
test_acc = accuracy_score(y_test, pred_test)
print ("train acc: {0:.2f}, test acc: {1:.2f}".format(train_acc, test_acc))

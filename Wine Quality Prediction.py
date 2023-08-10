#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')


# In[18]:


df = pd.read_csv('winequality.csv')
print(df.head())


# In[19]:


df.info()


# In[20]:


df.describe().T


# In[21]:


df.isnull().sum()


# In[22]:


for col in df.columns:
 if df[col].isnull().sum() > 0:
  df[col] = df[col].fillna(df[col].mean())

df.isnull().sum().sum()


# In[23]:


df = df.drop('total sulfur dioxide', axis=1)
df['best quality'] = [1 if x > 5 else 0 for x in df.quality]
features = df.drop(['quality', 'best quality'], axis=1)
target = df['best quality']

xtrain, xtest, ytrain, ytest = train_test_split(
	features, target, test_size=0.2, random_state=40)

xtrain.shape, xtest.shape


# In[24]:


#Training using Linear regression and SVM 
#Acurracy of both
norm = MinMaxScaler()
xtrain = norm.fit_transform(xtrain)
xtest = norm.transform(xtest)
models = [LogisticRegression(),SVC(kernel='rbf')]

for i in range(2):
	models[i].fit(xtrain, ytrain)

	print(f'{models[i]} : ')
	print('Training Accuracy : ', metrics.roc_auc_score(ytrain, models[i].predict(xtrain)))
	print('Validation Accuracy : ', metrics.roc_auc_score(
		ytest, models[i].predict(xtest)))
	print()


# In[25]:


#Confusion matrix
metrics.plot_confusion_matrix(models[1], xtest, ytest)
plt.show()


# In[26]:


print(metrics.classification_report(ytest,
									models[1].predict(xtest)))


# In[ ]:





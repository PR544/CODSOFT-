#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


# In[2]:


train_data = pd.read_csv('E:\\CS_DataSets\\Credit Card\\fraudTrain.csv')
test_data = pd.read_csv('E:\\CS_DataSets\\Credit Card\\fraudTest.csv')


# In[3]:


train_data.shape


# In[4]:


train_data.info()


# In[5]:


train_data.duplicated()


# In[6]:


train_data["is_fraud"].value_counts()


# In[7]:


train_data.describe()


# In[8]:


column_drop = ['Unnamed: 0','trans_date_trans_time','merchant','category','first','last','street','city','state','job','dob','trans_num','unix_time']
train_data.drop(columns=column_drop,inplace = True)
test_data.drop(columns=column_drop,inplace = True)


# In[9]:


train_data.nunique()


# In[10]:


print(train_data.shape)
print(test_data.shape)


# In[11]:


train_data['lat_dist'] = abs(round(train_data['merch_lat']-train_data['lat'],2))
train_data['long_dist'] = abs(round(train_data['merch_long']-train_data['long'],2))

test_data['lat_dist'] = abs(round(test_data['merch_lat']-test_data['lat'],2))
test_data['long_dist'] = abs(round(test_data['merch_long']-test_data['long'],2))


# In[12]:


round((train_data.isnull().sum()/train_data.shape[0])*100,2)


# In[13]:


train_data.gender =[ 1 if value == "M" else 0 for value in train_data.gender]
test_data.gender =[ 1 if value == "M" else 0 for value in test_data.gender]


# In[14]:


X_train = train_data.drop('is_fraud',axis=1)
X_test = test_data.drop('is_fraud',axis=1)
y_train = train_data['is_fraud']
y_test = test_data['is_fraud']


# In[15]:


scaler = StandardScaler()
X_trian = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# In[16]:


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state = 45)
dtc.fit(X_train,y_train)
y_pred = dtc.predict(X_test)


# In[17]:


accuracy = metrics.accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


# In[2]:


train_data = pd.read_csv('E:\CS_DataSets\Customer Churn\Churn_Modelling.csv')


# In[3]:


train_data.shape


# In[4]:


train_data.info()


# In[5]:


train_data.duplicated()


# In[6]:


train_data["Exited"].value_counts()


# In[7]:


train_data.describe()


# In[8]:


column_drop = ['RowNumber','CustomerId','Surname','Tenure','Exited','Geography','Gender']
X = train_data.drop(columns=column_drop,inplace = False)
y = train_data['Exited']


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[10]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[11]:


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)


# In[12]:


y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_rep)


# In[13]:


features = np.array([[661,35,150725.53,2,0,1,113656.85]])
prediction = model.predict(features)
print("Prediction: {}".format(prediction))


# In[ ]:





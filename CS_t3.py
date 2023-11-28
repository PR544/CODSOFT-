#!/usr/bin/env python
# coding: utf-8

# In[52]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


# In[53]:


sms = pd.read_csv('E:\CS_DataSets\spam.csv', encoding='latin-1')


# In[54]:


sms.head()


# In[55]:


sms=sms.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4"],axis=1)
sms=sms.rename(columns={"v1":"label","v2":"text"})
sms.head()


# In[56]:


sms.label.value_counts()


# In[57]:


sms.describe()


# In[58]:


sms['length']=sms['text'].apply(len)
sms.head()


# In[59]:


sms.loc[:,'label']=sms.label.map({'ham':0, 'spam':1})
sms.head()


# In[60]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

count=CountVectorizer()
input=['URGENT! You have won a 1 week FREE membership in our å£100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010 T&C www.dbuk.net LCCLTD POBOX 4403LDNW1A7RW18']

text=count.fit_transform(sms['text'])

x_train, x_test, y_train, y_test= train_test_split(text, sms['label'], test_size=0.20, random_state=1)
text


# In[61]:


print(x_train.shape)
print(x_test.shape)

input=text[5571]


# In[62]:


model=MLPClassifier()
model.fit(x_train, y_train)


# In[63]:


prediction=model.predict(x_test)
print(prediction)


# In[66]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("MLP Classifier")
print("Accuracy score: {}". format(accuracy_score(y_test, prediction)) )
print("Precision score: {}". format(precision_score(y_test, prediction)) )
print("Recall score: {}". format(recall_score(y_test, prediction)))
print("F1 score: {}". format(f1_score(y_test, prediction)))


# In[67]:


input
model.predict(input)


# In[ ]:





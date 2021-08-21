#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score


# # Load Dataset

# In[3]:


fire= pd.read_csv("forestfires.csv")
fire.head(10)


# # Preprocessing & Label Encoding 

# In[4]:


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
fire["month"] = label_encoder.fit_transform(fire["month"])
fire["day"] = label_encoder.fit_transform(fire["day"])
fire["size_category"] = label_encoder.fit_transform(fire["size_category"])


# In[5]:


fire.head(10)


# # Visualization

# In[15]:


for i in fire.describe().columns[:-2]:
    fire.plot.scatter(i,'area',grid=True)


# In[16]:


fire.groupby('day').area.mean().plot(kind='bar')


# In[17]:


fire.groupby('day').area.mean().plot(kind='box')


# In[18]:


fire.groupby('month').area.mean().plot(kind='box')


# In[19]:


fire.groupby('day').area.mean().plot(kind='line')


# In[20]:


fire.groupby('day').area.mean().plot(kind='hist')


# In[21]:


fire.groupby('day').area.mean().plot(kind='density')


# In[23]:


X=fire.iloc[:,:11]
X


# In[25]:


y=fire["size_category"]
y


# ## Split the Data

# In[26]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3)


# In[27]:


X_train.shape, y_train.shape, X_test.shape, y_test.shape


# # Grid Search CV

# In[28]:


clf = SVC()
param_grid = [{'kernel':['rbf'],'gamma':[50,5,10,0.5],'C':[15,14,13,12,11,10,0.1,0.001] }]
gsv = GridSearchCV(clf,param_grid,cv=10)
gsv.fit(X_train,y_train)


# In[29]:


gsv.best_params_ , gsv.best_score_


# In[30]:


clf = SVC(C= 15, gamma = 50)
clf.fit(X_train , y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred) * 100
print("Accuracy =", acc)
confusion_matrix(y_test, y_pred)


# In[31]:


clf1 = SVC(C= 15, gamma = 50)
clf1.fit(X , y)
y_pred = clf1.predict(X)
acc1 = accuracy_score(y, y_pred) * 100
print("Accuracy =", acc1)
confusion_matrix(y, y_pred)


# ### Poly
# 

# In[32]:


clf2 = SVC()
param_grid = [{'kernel':['poly'],'gamma':[50,5,10,0.5],'C':[15,14,13,12,11,10,0.1,0.001] }]
gsv = GridSearchCV(clf,param_grid,cv=10)
gsv.fit(X_train,y_train)


# In[33]:


gsv.best_params_ , gsv.best_score_


# ### Sigmoid
# 

# In[34]:


clf3 = SVC()
param_grid = [{'kernel':['sigmoid'],'gamma':[50,5,10,0.5],'C':[15,14,13,12,11,10,0.1,0.001] }]
gsv = GridSearchCV(clf,param_grid,cv=10)
gsv.fit(X_train,y_train)


# In[35]:


gsv.best_params_ , gsv.best_score_


# In[ ]:





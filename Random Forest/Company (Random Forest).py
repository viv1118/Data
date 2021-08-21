#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import preprocessing


# # Load Dataset

# In[4]:


company=pd.read_csv("company_data.csv")
company.head(10)


# In[5]:


company.shape


# In[8]:


company.dtypes


# In[9]:


company.info()


# # Converting from Categorical Data

# In[10]:


company['High'] = company.Sales.map(lambda x: 1 if x>8 else 0)


# In[11]:


company['ShelveLoc']=company['ShelveLoc'].astype('category')


# In[12]:


company['Urban']=company['Urban'].astype('category')


# In[13]:


company['US']=company['US'].astype('category')


# In[14]:


company.dtypes


# In[15]:


company.head(10)


# # label encoding to convert categorical values into numeric.

# In[16]:


company['ShelveLoc']=company['ShelveLoc'].cat.codes


# In[17]:


company['Urban']=company['Urban'].cat.codes


# In[18]:


company['US']=company['US'].cat.codes


# In[20]:


company.head(10)


# In[21]:


company.tail(10)


# # Visualization

# In[23]:


sns.pairplot(company)


# In[24]:


sns.barplot(company['Sales'], company['Income'])


# In[25]:


sns.boxplot(company['Sales'], company['Income'])


# In[26]:


sns.lmplot(x='Income', y='Sales', data=company)


# In[27]:


sns.jointplot(company['Sales'], company['Income'])


# In[28]:


sns.stripplot(company['Sales'], company['Income'])


# In[29]:


sns.distplot(company['Sales'])


# In[30]:


sns.distplot(company['Income'])


# # setting feature and target variables

# In[31]:


feature_cols=['CompPrice','Income','Advertising','Population','Price','ShelveLoc','Age','Education','Urban','US']


# In[32]:


x = company.drop(['Sales', 'High'], axis = 1)


# In[33]:


x = company[feature_cols]


# In[34]:


y = company.High


# In[35]:


print(x)


# In[36]:


print(y)


# # Splitting the dataset into the Training set and Test set

# In[37]:


from sklearn.model_selection import train_test_split


# In[38]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


# In[39]:


print(x_train)


# In[40]:


print(y_train)


# In[41]:


print(x_test)


# In[42]:


print(y_test)


# # Feature Scaling
# 

# In[43]:


from sklearn.preprocessing import StandardScaler


# In[44]:


sc = StandardScaler()


# In[45]:


x_train = sc.fit_transform(x_train)


# In[46]:


x_test = sc.transform(x_test)


# In[47]:


print(x_train)


# In[48]:


print(x_test)


# # Training the Random Forest Classification model on the Training set

# In[53]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)


# In[60]:


classifier.fit(x_train, y_train)
 


# In[61]:


classifier.score(x_test, y_test)


# # Predicting the Test set results

# In[63]:


y_pred = classifier.predict(x_test)


# In[65]:


y_pred


# # Making the Confusion Matrix

# In[66]:


from sklearn.metrics import confusion_matrix, accuracy_score


# In[67]:


cm = confusion_matrix(y_test, y_pred)


# In[68]:


print(cm)


# In[69]:


accuracy_score(y_test, y_pred)


# In[77]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100, criterion='gini')
classifier.fit(x_train, y_train)


# In[78]:


classifier.score(x_test, y_test)


# In[ ]:





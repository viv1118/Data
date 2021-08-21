#!/usr/bin/env python
# coding: utf-8

# # Import Libraries 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Load Dataset

# In[2]:


salary_train=pd.read_csv('SalaryData_Train.csv')


# In[3]:


salary_test=pd.read_csv('SalaryData_Test.csv')


# In[4]:


salary_train.columns


# In[5]:


salary_test.columns


# In[6]:


salary_train.shape


# In[7]:


salary_test.shape


# In[8]:


salary_train.dtypes


# In[9]:


salary_test.dtypes


# In[10]:


salary_train.info()


# In[11]:


salary_test.info()


# In[12]:


string_columns=['workclass','education','maritalstatus','occupation','relationship','race','sex','native']


# # Visualization

# In[13]:


sns.pairplot(salary_train)


# In[14]:


sns.pairplot(salary_test)


# In[16]:


sns.boxplot(salary_train['Salary'], salary_train['capitalgain'])


# In[17]:


sns.boxplot(salary_test['Salary'], salary_test['capitalgain'])


# In[19]:


sns.countplot(salary_train['Salary'])


# In[20]:


sns.countplot(salary_test['Salary'])


# In[22]:


plt.figure(figsize=(20,10))
sns.barplot(x='Salary', y='hoursperweek', data=salary_train)
plt.show()


# In[23]:


plt.figure(figsize=(20,10))
sns.barplot(x='Salary', y='hoursperweek', data=salary_test)
plt.show()


# In[32]:


sns.distplot(salary_train['hoursperweek'])


# In[33]:


sns.distplot(salary_test['hoursperweek'])


# In[41]:


plt.figure(figsize=(15,10))
sns.lmplot(y='capitalgain', x='hoursperweek',data=salary_train)
plt.show()


# In[42]:


plt.figure(figsize=(15,10))
sns.lmplot(y='capitalgain', x='hoursperweek',data=salary_test)
plt.show()


# In[46]:


sns.factorplot(x='Salary', hue='hoursperweek', data=salary_train, kind='count', height=10,aspect=1.5)


# In[47]:


sns.factorplot(x='Salary', hue='hoursperweek', data=salary_test, kind='count', height=10,aspect=1.5)


# In[48]:


sns.catplot(x='Salary', hue='hoursperweek', data=salary_train, kind='count',aspect=1.5)


# In[50]:


sns.catplot(x='Salary', hue='hoursperweek', data=salary_test, kind='count',aspect=1.5)


# In[55]:


plt.subplots(1,2, figsize=(16,8))

colors = ["#FA5858", "#64FE2E"]
labels ="capitalgain", "capitalloss"

plt.suptitle('salary of an individual', fontsize=20)

salary_train["Salary"].value_counts().plot.pie(explode=[0,0.25], autopct='%1.2f%%', shadow=True, colors=colors, 
                                             labels=labels, fontsize=12, startangle=25)


# In[53]:


plt.style.use('seaborn-whitegrid')

salary_train.hist(bins=20, figsize=(15,10), color='red')
plt.show()


# In[56]:


plt.subplots(1,2, figsize=(16,8))

colors = ["#FA5858", "#64FE2E"]
labels ="capitalgain", "capitalloss"

plt.suptitle('salary of an individual', fontsize=20)

salary_test["Salary"].value_counts().plot.pie(explode=[0,0.25], autopct='%1.2f%%', shadow=True, colors=colors, 
                                             labels=labels, fontsize=12, startangle=25)


# In[57]:


plt.style.use('seaborn-whitegrid')

salary_test.hist(bins=20, figsize=(15,10), color='red')
plt.show()


# # Preprocessing

# In[58]:


from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()


# In[59]:


for i in string_columns:
    salary_train[i]=label_encoder.fit_transform(salary_train[i])
    salary_test[i]=label_encoder.fit_transform(salary_test[i])


# In[61]:


col_names=list(salary_train.columns)
col_names


# In[63]:


train_X=salary_train[col_names[0:13]]
train_X


# In[65]:


train_Y=salary_train[col_names[13]]
train_Y


# In[67]:


test_x=salary_test[col_names[0:13]]
test_x


# In[69]:


test_y=salary_test[col_names[13]]
test_y


# # Building a Model of Naive Bayes

# # Gaussian Naive Bayes

# In[70]:


from sklearn.naive_bayes import GaussianNB
Gnbmodel=GaussianNB()


# In[72]:


train_pred_gau=Gnbmodel.fit(train_X,train_Y).predict(train_X)
train_pred_gau


# In[74]:


test_pred_gau=Gnbmodel.fit(train_X,train_Y).predict(test_x)
test_pred_gau


# In[75]:


train_acc_gau=np.mean(train_pred_gau==train_Y)


# In[76]:


test_acc_gau=np.mean(test_pred_gau==test_y)


# In[78]:


train_acc_gau


# In[79]:


test_acc_gau


# # Multinomial Naive Bayes
# 

# In[80]:


from sklearn.naive_bayes import MultinomialNB
Mnbmodel=MultinomialNB()


# In[83]:


train_pred_multi=Mnbmodel.fit(train_X,train_Y).predict(train_X)
train_pred_multi


# In[84]:


test_pred_multi=Mnbmodel.fit(train_X,train_Y).predict(test_x)
test_pred_multi


# In[86]:


train_acc_multi=np.mean(train_pred_multi==train_Y)
train_acc_multi


# In[88]:


test_acc_multi=np.mean(test_pred_multi==test_y)
test_acc_multi


# In[ ]:





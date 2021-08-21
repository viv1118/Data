#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn import preprocessing


# # Load Dataset

# In[3]:


fraud=pd.read_csv("Fraud_check.csv")
fraud.head(10)


# In[4]:


fraud.dtypes


# In[5]:


fraud.info()


# In[7]:


fraud.columns


# In[8]:


fraud.shape


# In[9]:


fraud.isnull().sum()


# In[11]:


fraud["TaxInc"] = pd.cut(fraud["Taxable.Income"], bins = [10002,30000,99620], labels = ["Risky", "Good"])
fraud["TaxInc"]


# In[13]:


fraudcheck = fraud.drop(columns=["Taxable.Income"])
fraudcheck 


# In[14]:


FC = pd.get_dummies(fraudcheck .drop(columns = ["TaxInc"]))


# In[15]:


Fraud_final = pd.concat([FC,fraudcheck ["TaxInc"]], axis = 1)


# In[17]:


colnames = list(Fraud_final.columns)
colnames


# In[19]:


predictors = colnames[:9]
predictors


# In[21]:


target = colnames[9]
target


# In[24]:


X = Fraud_final[predictors]
X.shape


# In[26]:


Y = Fraud_final[target]
Y


# # Visualization

# In[27]:


sns.pairplot(fraud)


# In[28]:


sns.barplot(fraud['Taxable.Income'], fraud['City.Population'])


# In[29]:


sns.boxplot(fraud['Taxable.Income'], fraud['City.Population'])


# In[30]:


sns.lmplot(x='Taxable.Income',y='City.Population', data=fraud)


# In[31]:


sns.jointplot(fraud['Taxable.Income'], fraud['City.Population'])


# In[33]:


sns.stripplot(fraud['Taxable.Income'], fraud['City.Population'])


# In[34]:


sns.distplot(fraud['Taxable.Income'])


# In[35]:


sns.distplot(fraud['City.Population'])


# # Building Random Forest Model

# In[36]:


from sklearn.ensemble import RandomForestClassifier


# In[37]:


rf = RandomForestClassifier(n_jobs = 3, oob_score = True, n_estimators = 15, criterion = "entropy")


# In[38]:


rf = RandomForestClassifier(n_jobs=3,oob_score=True,n_estimators=15,criterion="entropy")


# In[41]:


np.shape(Fraud_final)  


# In[47]:


Fraud_final.describe()


# In[48]:


Fraud_final.info()


# In[49]:


type([X])


# In[50]:


type([Y])


# In[52]:


Y1 = pd.DataFrame(Y)
Y1


# In[53]:


type(Y1)


# In[54]:


rf.fit(X,Y1) 


# In[55]:


rf.estimators_ 


# In[56]:


rf.classes_ 


# In[57]:


rf.n_classes_  


# In[58]:


rf.n_features_  


# In[59]:


rf.n_outputs_ 


# In[60]:


rf.oob_score_  


# In[61]:


rf.predict(X)


# In[62]:


Fraud_final['rf_pred'] = rf.predict(X)


# In[63]:


cols = ['rf_pred','TaxInc']


# In[64]:


Fraud_final[cols].head()


# In[65]:


Fraud_final["TaxInc"]


# In[66]:


from sklearn.metrics import confusion_matrix


# In[67]:


confusion_matrix(Fraud_final['TaxInc'],Fraud_final['rf_pred']) # Confusion matrix


# In[68]:


pd.crosstab(Fraud_final['TaxInc'],Fraud_final['rf_pred'])


# In[69]:


print("Accuracy",(476+115)/(476+115+9+0)*100)


# In[70]:


Fraud_final["rf_pred"]


# In[ ]:





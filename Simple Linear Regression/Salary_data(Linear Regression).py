#!/usr/bin/env python
# coding: utf-8

# # Import Library
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
import seaborn as sns


# # load Data 

# In[2]:


data=pd.read_csv("Salary_Data.csv")
data


# In[3]:


data.head()


# In[4]:


data.info()


# # Graphical Representation of Data

# In[5]:


data.plot()


# In[6]:


data.corr()


# In[7]:


data.Salary


# In[8]:


data.YearsExperience


# In[9]:


sns.distplot(data['Salary'])


# In[10]:


sns.distplot(data['YearsExperience'])


# In[11]:


sns.pairplot(data)


# In[12]:


sns.scatterplot(x=data.YearsExperience, y=np.log(data.Salary), data=data)


# # Calculate R^2 values

# In[16]:


import statsmodels.formula.api as smf
import pandas.util.testing as tm
model = smf.ols("Salary~YearsExperience",data = data).fit()


# In[17]:


sns.regplot(x="Salary", y="YearsExperience", data=data);


# In[18]:


#Coefficients
model.params


# In[19]:


model =smf.ols('Salary~YearsExperience', data=data).fit()
model


# In[20]:


model.summary()


# # Predict for new data point

# In[21]:


#Predict for 15 and 20 Year's of Experiance 
newdata=pd.Series([15,20])


# In[23]:


data_pred=pd.DataFrame(newdata,columns=['YearsExperience'])
data_pred


# In[24]:


model.predict(data_pred).round(2)


# In[ ]:





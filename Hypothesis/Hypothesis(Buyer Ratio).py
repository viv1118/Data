#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import scipy as sp
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.proportion import proportions_ztest


# # Load Dataset

# In[4]:


BuyerRatio =pd.read_csv('BuyerRatio (1).csv')
BuyerRatio.head(10)


# In[5]:


BuyerRatio.shape


# In[6]:


BuyerRatio.dtypes


# In[7]:


BuyerRatio.info()


# In[8]:


BuyerRatio.describe()


# In[10]:


East=BuyerRatio['East'].mean()
print('East Mean = ',East)


# In[12]:


West=BuyerRatio['West'].mean()
print('West Mean = ',West)


# In[14]:


North=BuyerRatio['North'].mean()
print('North Mean = ',North)


# In[16]:


South=BuyerRatio['South'].mean()
print('South Mean = ',South)


# In[ ]:


#The Null and Alternative Hypothesis

There are no significant differences between the groups' mean values. H0:μ1=μ2=μ3=μ4=μ5

There is a significant difference between the groups' mean values. Ha:μ1≠μ2≠μ3≠μ4


# # Visualization

# In[17]:


sns.distplot(BuyerRatio['East'])


# In[18]:


sns.distplot(BuyerRatio['West'])


# In[19]:


sns.distplot(BuyerRatio['North'])


# In[20]:


sns.distplot(BuyerRatio['South'])


# In[21]:


sns.distplot(BuyerRatio['East'])
sns.distplot(BuyerRatio['West'])
sns.distplot(BuyerRatio['North'])
sns.distplot(BuyerRatio['South'])
plt.legend(['East','West','North','South'])


# In[22]:


sns.boxplot(data=[BuyerRatio['East'],BuyerRatio['West'],BuyerRatio['North'],BuyerRatio['South']],notch=True)
plt.legend(['East','West','North','South'])


# # Hypothesis Testing

# In[23]:


alpha=0.05
Male = [50,142,131,70]
Female=[435,1523,1356,750]
Sales=[Male,Female]
print(Sales)


# In[25]:


chiStats = sp.stats.chi2_contingency(Sales)
print('Test t=%f p-value=%f' % (chiStats[0], chiStats[1]))
print('Interpret by p-Value')


# In[26]:


if chiStats[1] < 0.05:
  print('we reject null hypothesis')
else:
  print('we accept null hypothesis')


# # Critical Value

# In[28]:


alpha = 0.05
critical_value = sp.stats.chi2.ppf(q = 1 - alpha,df=chiStats[2])
critical_value 


# # Degree of Freedom

# In[29]:


observed_chi_val = chiStats[0]
print('Interpret by critical value')


# In[30]:


if observed_chi_val <= critical_value:
    print ('Null hypothesis cannot be rejected (variables are not related)')
else:
    print ('Null hypothesis cannot be excepted (variables are not independent)')


# In[ ]:


#Inference : proportion of male and female across regions is same


# In[ ]:





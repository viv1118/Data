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

# In[5]:


data1 = pd.read_csv("Cutlets (1).csv")
data1.head(10)


# In[6]:


data1.shape


# In[7]:


data1.dtypes


# In[8]:


data1.info()


# In[9]:


data1.describe(include='all')


# In[10]:


Unit_A=data1['Unit A'].mean()


# In[11]:


Unit_B=data1['Unit B'].mean()


# In[12]:


print('Unit A Mean = ',Unit_A, '\nUnit B Mean = ',Unit_B)


# In[13]:


print('Unit A Mean > Unit B Mean = ',Unit_A>Unit_B)


# # Visualization

# In[14]:


sns.distplot(data1['Unit A'])


# In[15]:


sns.distplot(data1['Unit B'])


# In[17]:


sns.distplot(data1['Unit A'])
sns.distplot(data1['Unit B'])
plt.legend(['Unit A','Unit B'])


# In[18]:


sns.boxplot(data=[data1['Unit A'],data1['Unit B']],notch=True)
plt.legend(['Unit A','Unit B'])


# # Hypothesis Testing

# In[22]:


alpha=0.05
UnitA=pd.DataFrame(data1['Unit A'])
UnitA


# In[23]:


UnitB=pd.DataFrame(data1['Unit B'])
UnitB


# In[21]:


print(UnitA,UnitB)


# In[24]:


tStat,pValue =sp.stats.ttest_ind(UnitA,UnitB)


# In[25]:


print("P-Value:{0} T-Statistic:{1}".format(pValue,tStat))


# In[26]:


if pValue <0.05:
  print('we reject null hypothesis')
else:
  print('we accept null hypothesis')


# In[ ]:


#Inference is that there is no significant difference in the diameters of Unit A and Unit B


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


#import the libraries
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


# # Import Dataset

# In[3]:


Fantaloons=pd.read_csv('Faltoons (1).csv')
Fantaloons.head(10)


# In[4]:


Fantaloons.shape


# In[5]:


Fantaloons.dtypes


# In[6]:


Fantaloons.info()


# In[7]:


Fantaloons.describe()


# In[8]:


Weekdays_value=Fantaloons['Weekdays'].value_counts()


# In[9]:


Weekend_value=Fantaloons['Weekend'].value_counts()


# In[10]:


print(Weekdays_value,Weekend_value)


# # Hypothesis Testing

# In[13]:


tab = Fantaloons.groupby(['Weekdays', 'Weekend']).size()
tab


# In[14]:


count = np.array([280, 520]) 
count


# In[16]:


nobs = np.array([400, 400])
nobs


# In[17]:


stat, pval = proportions_ztest(count, nobs,alternative='two-sided') 


# In[18]:


print('{0:0.3f}'.format(pval))


# In[19]:


stat, pval = proportions_ztest(count, nobs,alternative='larger')


# In[20]:


print('{0:0.3f}'.format(pval))


# In[ ]:


#P-value <0.05 and hence we reject null. We reject null Hypothesis. Hence proportion of Female is greater than Male


# In[ ]:





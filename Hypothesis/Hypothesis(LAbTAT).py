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

# In[3]:


LabTAT =pd.read_csv('LabTAT (1).csv')
LabTAT.head(10)


# In[4]:


LabTAT.shape


# In[5]:


LabTAT.dtypes


# In[6]:


LabTAT.info()


# In[7]:


LabTAT.describe()


# In[9]:


Laboratory_1=LabTAT['Laboratory 1'].mean()
print('Laboratory 1 Mean = ',Laboratory_1)


# In[11]:


Laboratory_2=LabTAT['Laboratory 2'].mean()
print('Laboratory 2 Mean = ',Laboratory_2)


# In[13]:


Laboratory_3=LabTAT['Laboratory 3'].mean()
print('Laboratory 3 Mean = ',Laboratory_3)


# In[15]:


Laboratory_4=LabTAT['Laboratory 4'].mean()
print('Laboratory 4 Mean = ',Laboratory_4)


# In[16]:


print('Laboratory_1 > Laboratory_2 = ',Laboratory_1 > Laboratory_2)
print('Laboratory_2 > Laboratory_3 = ',Laboratory_2 > Laboratory_3)
print('Laboratory_3 > Laboratory_4 = ',Laboratory_3 > Laboratory_4)
print('Laboratory_4 > Laboratory_1 = ',Laboratory_4 > Laboratory_1)


# In[ ]:


#The Null and Alternative Hypothesis

There are no significant differences between the groups' mean Lab values. H0:μ1=μ2=μ3=μ4

There is a significant difference between the groups' mean Lab values. Ha:μ1≠μ2≠μ3≠μ4


# # Visualization

# In[17]:


sns.distplot(LabTAT['Laboratory 1'])


# In[18]:


sns.distplot(LabTAT['Laboratory 2'])


# In[19]:


sns.distplot(LabTAT['Laboratory 3'])


# In[20]:


sns.distplot(LabTAT['Laboratory 4'])


# In[21]:


sns.distplot(LabTAT['Laboratory 1'])
sns.distplot(LabTAT['Laboratory 2'])
sns.distplot(LabTAT['Laboratory 3'])
sns.distplot(LabTAT['Laboratory 4'])
plt.legend(['Laboratory 1','Laboratory 2','Laboratory 3','Laboratory 4'])


# In[22]:


sns.boxplot(data=[LabTAT['Laboratory 1'],LabTAT['Laboratory 2'],LabTAT['Laboratory 3'],LabTAT['Laboratory 4']],notch=True)
plt.legend(['Laboratory 1','Laboratory 2','Laboratory 3','Laboratory 4'])


# # Hypothesis Testing

# In[24]:


alpha=0.05
Lab1=pd.DataFrame(LabTAT['Laboratory 1'])
Lab1


# In[26]:


Lab2=pd.DataFrame(LabTAT['Laboratory 2'])
Lab2


# In[28]:


Lab3=pd.DataFrame(LabTAT['Laboratory 3'])
Lab3


# In[30]:


Lab4=pd.DataFrame(LabTAT['Laboratory 4'])
Lab4


# In[31]:


print(Lab1,Lab1,Lab3,Lab4)


# In[32]:


tStat, pvalue = sp.stats.f_oneway(Lab1,Lab2,Lab3,Lab4)


# In[34]:


print("P-Value:{0} T-Statistic:{1}".format(pvalue,tStat))


# In[36]:


if pvalue < 0.05:
  print('we reject null hypothesis')
else:
  print('we accept null hypothesis')


# In[ ]:


#Inference is that there no significant difference in the average TAT for all the labs.


# In[ ]:





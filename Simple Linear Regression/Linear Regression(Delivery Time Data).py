#!/usr/bin/env python
# coding: utf-8

# # Import Library¶

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
import seaborn as sns
from scipy.stats import kurtosis
from scipy.stats import skew


# # Load Data

# In[2]:


data1=pd.read_csv("delivery_time.csv")
data1


# In[3]:


data1.head()


# In[4]:


data1.describe()


# In[5]:


data1.info()


# In[6]:


data1.shape


# In[7]:


data1 = data1.rename(columns = {'Delivery Time': 'DT', 'Sorting Time': 'ST'}, inplace = False)
data1.info()


# In[8]:


print(kurtosis(data1.DT))


# In[9]:


print(kurtosis(data1.ST))


# In[10]:


print(skew(data1.DT))


# In[11]:


print(skew(data1.ST))


# # Graphical Representation of Data

# In[12]:


data1.plot()


# In[13]:


sns.pairplot(data1)


# In[14]:


data1.corr()


# In[15]:


sns.distplot(data1['DT'])


# In[16]:


sns.distplot(data1['ST'])


# In[17]:


corrMatrix = data1.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()


# In[18]:


cols = data1.columns 
colours = ['#ffc0cb', '#ffff00']
sns.heatmap(data1[cols].isnull(),
            cmap=sns.color_palette(colours))


# In[19]:


data1.boxplot(column=['DT'])


# In[20]:


data1.boxplot(column=['ST'])


# In[21]:


data1[data1.duplicated()].shape


# In[22]:


data1['ST'].hist()


# In[23]:


data1.boxplot(column=['ST'])


# In[24]:


data1['ST'].value_counts().plot.bar()


# # calculate R^2 values

# In[25]:


import statsmodels.formula.api as smf
model = smf.ols("DT~ST",data = data1).fit()
model


# In[26]:


sns.regplot(x="ST", y="DT", data=data1);


# In[27]:


model.params


# In[28]:


print(model.tvalues, '\n', model.pvalues) 


# In[29]:


(model.rsquared,model.rsquared_adj)


# In[30]:


model.summary()


# In[31]:


data_1=data1
data_1['DT'] = np.log(data_1['DT'])
data_1['ST'] = np.log(data_1['ST'])
sns.distplot(data_1['DT'])
fig = plt.figure()
sns.distplot(data_1['ST'])
fig = plt.figure()


# In[32]:


model_2 = smf.ols("ST~DT",data = data_1).fit()
model_2.summary()


# In[33]:


data_2=data1
data_1['DT'] = np.log(data_1['DT'])
sns.distplot(data_1['DT'])
fig = plt.figure()
sns.distplot(data_1['ST'])
fig = plt.figure()


# In[34]:


model_3 = smf.ols("ST~DT",data = data_2).fit()
model_3.summary()


# In[35]:


data_3=data1
data_1['ST'] = np.log(data_1['ST'])
sns.distplot(data_1['DT'])
fig = plt.figure()
sns.distplot(data_1['ST'])
fig = plt.figure()


# In[36]:


model_4 = smf.ols("ST~DT",data = data_3).fit()
model_4.summary()


# In[45]:


import statsmodels.formula.api as smf
import numpy as np
import pandas.util.testing as tm
model_4 = smf.ols("DT~ST",data = data_3).fit()


# # Predict for new data point

# In[46]:


#Predict for 15min and 20min Sorting Time
newdata=pd.Series([15,20])


# In[47]:


data_pred=pd.DataFrame(newdata,columns=['ST'])
data_pred


# In[48]:


model_4.predict(data_pred)


# In[ ]:





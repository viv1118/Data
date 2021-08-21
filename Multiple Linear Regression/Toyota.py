#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from statsmodels.graphics.regressionplots import influence_plot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


# # Load Data

# In[3]:


Toyota = pd.read_csv("ToyotaCorolla.csv",encoding='ISO-8859-1')
Toyota


# In[10]:


Toyota1= Toyota.iloc[:,[2,3,6,8,12,13,15,16,17]]
Toyota1


# In[12]:


Toyota1.rename(columns={"Age_08_04":"Age"},inplace=True)


# In[9]:


eda=Toyota1.describe()
eda


# # Data Visualization

# In[14]:


plt.boxplot(Toyota1["Price"])


# In[15]:


plt.boxplot(Toyota1["Age"])


# In[16]:


plt.boxplot(Toyota1["HP"])


# In[17]:


plt.boxplot(Toyota1["cc"])


# In[18]:


plt.boxplot(Toyota1["Quarterly_Tax"])


# In[19]:


plt.boxplot(Toyota1["Weight"])


# In[22]:


plt.scatter(Toyota1['Age'], Toyota1['Price'], c = 'red')
plt.title('Price vs Age of the Cars')
plt.xlabel('Age in Years')
plt.ylabel('Price(Euros)')
plt.show()


# In[23]:


plt.figure(figsize=(8,8))
plt.title('Car Price Distribution Plot')
sns.distplot(Toyota1['Price'])


# In[24]:


import statsmodels.api as sm


# In[25]:


sm.graphics.qqplot(Toyota1["Price"])


# In[26]:


sm.graphics.qqplot(Toyota1["Age"])


# In[27]:


sm.graphics.qqplot(Toyota1["HP"])


# In[28]:


sm.graphics.qqplot(Toyota1["Quarterly_Tax"])


# In[29]:


sm.graphics.qqplot(Toyota1["Weight"])


# In[30]:


sm.graphics.qqplot(Toyota1["Gears"])


# In[31]:


sm.graphics.qqplot(Toyota1["Doors"])


# In[32]:


sm.graphics.qqplot(Toyota1["cc"])


# In[33]:


plt.hist(Toyota1["Price"])


# In[34]:


plt.hist(Toyota1["Age"])


# In[35]:


plt.hist(Toyota1["HP"])


# In[36]:


plt.hist(Toyota1["Quarterly_Tax"])


# In[37]:


plt.hist(Toyota1["Weight"])


# In[38]:


sns.pairplot(Toyota1)


# In[41]:


plt.hist(Toyota1['KM'], edgecolor = 'white', bins = 5)
plt.title('Histogram of Kilometer')
plt.xlabel('Kilometer')
plt.ylabel('Frequency')
plt.show()


# In[42]:


plt.figure(figsize=(20, 6))
plt.hist(Toyota1['KM'],facecolor ="peru",edgecolor ="blue",bins =100)
plt.ylabel("Frequency");
plt.xlabel(" Total KM")
plt.show()


# In[43]:


plt.figure(figsize=(20, 6))
plt.hist(Toyota1['Weight'],facecolor ="yellow",edgecolor ="blue",bins =15)
plt.ylabel("Frequency");
plt.xlabel(" Total Weight")
plt.show()


# In[45]:


fuel_count = pd.value_counts(Toyota1['cc'].values, sort = True)
plt.xlabel('Frequency')
plt.ylabel('cc')
plt.title('Bar plot of cc')
fuel_count.plot.barh()


# In[46]:


sns.set(style = 'darkgrid')
sns.regplot(x = Toyota1['Age'], y = Toyota1['Price'], marker = '*')


# # building on individual model

# In[49]:


correlation_values= Toyota1.corr()
correlation_values


# In[50]:


import statsmodels.formula.api as smf


# In[51]:


m1= smf.ols("Price~Age+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight",data= Toyota1).fit()


# In[52]:


m1.summary()


# In[53]:


m1_cc = smf.ols("Price~cc",data= Toyota1).fit()


# In[54]:


m1_cc.summary()


# In[55]:


m1_doors = smf.ols("Price~Doors", data= Toyota1).fit()


# In[56]:


m1_doors.summary()


# In[57]:


m1_to = smf.ols("Price~cc+Doors",data= Toyota1).fit()


# In[58]:


m1_to.summary()


# In[59]:


import statsmodels.api as sm


# In[60]:


sm.graphics.influence_plot(m1)


# In[61]:


Toyota2= Toyota1.drop(Toyota.index[[80]],axis=0)


# In[62]:


m2= smf.ols("Price~Age+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight",data= Toyota2).fit()


# In[63]:


m2.summary()


# In[64]:


Toyota3 = Toyota1.drop(Toyota.index[[80,221]],axis=0)


# In[65]:


m3= smf.ols("Price~Age+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight",data= Toyota3).fit()


# In[66]:


m3.summary()


# In[67]:


Toyota4= Toyota1.drop(Toyota.index[[80,221,960]],axis=0)


# In[68]:


m4= smf.ols("Price~Age+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight", data = Toyota4).fit()


# In[69]:


m4.summary()


# # Final Modal

# In[70]:


Finalmodel = smf.ols("Price~Age+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight", data = Toyota4).fit()


# In[71]:


Finalmodel.summary()


# # Prediction

# In[73]:


Finalmodel_pred = Finalmodel.predict(Toyota4)
Finalmodel_pred


# # Validation 

# In[74]:


plt.scatter(Toyota4["Price"],Finalmodel_pred,c='r');plt.xlabel("Observed values");plt.ylabel("Predicted values")


# # Residuals v/s Fitted values

# In[75]:


plt.scatter(Finalmodel_pred, Finalmodel.resid_pearson,c='r');plt.axhline(y=0,color='blue');plt.xlabel("Fitted values");plt.ylabel("Residuals")


# In[76]:


plt.hist(Finalmodel.resid_pearson) 


# # QQ Plot

# In[77]:


import pylab


# In[78]:


import scipy.stats as st


# In[79]:


st.probplot(Finalmodel.resid_pearson, dist='norm',plot=pylab)


# # Testing of Final Model

# In[80]:


from sklearn.model_selection import train_test_split


# In[81]:


train_data,test_Data= train_test_split(Toyota1,test_size=0.3)


# In[82]:


Finalmodel1 = smf.ols("Price~Age+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight", data = train_data).fit()


# In[83]:


Finalmodel1.summary()


# In[85]:


Finalmodel_pred = Finalmodel1.predict(train_data)
Finalmodel_pred


# # Training and Testing of Residual Data

# In[87]:


Finalmodel_res = train_data["Price"]-Finalmodel_pred
Finalmodel_res


# In[89]:


Finalmodel_rmse = np.sqrt(np.mean(Finalmodel_res*Finalmodel_res))
Finalmodel_rmse


# In[91]:


Finalmodel_testpred = Finalmodel1.predict(test_Data)
Finalmodel_testpred


# In[93]:


Finalmodel_testres= test_Data["Price"]-Finalmodel_testpred
Finalmodel_testres


# In[95]:


Finalmodel_testrmse = np.sqrt(np.mean(Finalmodel_testres*Finalmodel_testres))
Finalmodel_testrmse


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import math 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


# # Importing the dataset

# In[3]:


startups = pd.read_csv('50_Startups.csv')
startups


# In[4]:


len(startups)


# In[5]:


startups.head()


# In[6]:


startups.shape


# In[8]:


plt.scatter(startups['Marketing Spend'], startups['Profit'], alpha=0.5)
plt.title('Scatter plot of Profit with Marketing Spend')
plt.xlabel('Marketing Spend')
plt.ylabel('Profit')
plt.show()


# In[9]:


plt.scatter(startups['R&D Spend'], startups['Profit'], alpha=0.5)
plt.title('Scatter plot of Profit with R&D Spend')
plt.xlabel('R&D Spend')
plt.ylabel('Profit')
plt.show()


# In[10]:


plt.scatter(startups['Administration'], startups['Profit'], alpha=0.5)
plt.title('Scatter plot of Profit with Administration')
plt.xlabel('Administration')
plt.ylabel('Profit')
plt.show()


# # Create The Figure  Object

# In[12]:


ax=startups.groupby(['State'])['Profit'].mean().plot.bar(figsize=(10,5), fontsize=14)


# In[14]:


ax=sns.pairplot(startups)


# In[20]:


startups.State.value_counts()


# # Create Dummy Variables for the categorical variables States

# In[22]:


startups['New York_State']= np.where(startups['State']=='New York',1,0)


# In[23]:


startups['California_State']= np.where(startups['State']=='California',1,0)


# In[24]:


startups['Florida_State']= np.where(startups['State']=='Florida',1,0)


# # Drop the original column State from the Dataframe

# In[26]:


startups.drop(columns=['State'], axis=1, inplace=True)


# In[27]:


startups.head()


# In[28]:


dependent_variable='Profit'


# # Create a list of independent Variables

# In[29]:


independent_variables=startups.columns.tolist()


# In[30]:


independent_variables.remove(dependent_variable)


# In[31]:


independent_variables


# # Create the data of Independent variables

# In[33]:


X=startups[independent_variables].values
X


# # Create the Dependent Variables Data

# In[36]:


y=startups[dependent_variable].values
y


# # Splitting the dataset into the Training set and Test set

# In[37]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# # Transforming Data

# In[39]:


scaler=MinMaxScaler()


# In[40]:


X_train=scaler.fit_transform(X_train)


# In[41]:


X_test=scaler.transform(X_test)


# In[42]:


X_train[0:10]


# # Training the Multiple Linear Regression model on the Training set

# In[43]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# # Predicting the Test set results

# In[44]:


y_pred = regressor.predict(X_test)


# In[45]:


math.sqrt(mean_squared_error(y_test, y_pred))


# In[46]:


r2_score(y_test, y_pred)


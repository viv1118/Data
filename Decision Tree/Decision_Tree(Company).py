#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree 
from sklearn.metrics import classification_report 
from sklearn import preprocessing


# # Load Dataset

# In[3]:


Comp_Data= pd.read_csv("Company_Data.csv")
Comp_Data.head(20)


# In[4]:


Comp_Data.shape


# In[5]:


Comp_Data.info()


# In[6]:


Comp_Data.dtypes


# In[7]:


Comp_Data.describe()


# In[8]:


Comp_Data.corr()


# # Visualization

# In[11]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.pairplot(Comp_Data)


# In[12]:


sns.barplot(Comp_Data['Sales'], Comp_Data['Income'])


# In[13]:


sns.boxplot(Comp_Data['Sales'], Comp_Data['Income'])


# In[14]:


sns.lmplot(x='Income', y='Sales', data=Comp_Data)


# In[15]:


sns.jointplot(Comp_Data['Sales'], Comp_Data['Income'])


# In[17]:


sns.swarmplot(Comp_Data['Sales'], Comp_Data['Income'])


# In[19]:


sns.distplot(Comp_Data['Sales'])


# In[20]:


sns.distplot(Comp_Data['Income'])


# ## Preprocessing

# In[21]:


Comp_Data.loc[Comp_Data["Sales"] <= 10.00,"Sales1"]="Not High"
Comp_Data.loc[Comp_Data["Sales"] >= 10.01,"Sales1"]="High"


# In[22]:


Comp_Data


# ## Label Encoding

# In[23]:


label_encoder = preprocessing.LabelEncoder()
Comp_Data["ShelveLoc"] = label_encoder.fit_transform(Comp_Data["ShelveLoc"])
Comp_Data["Urban"] = label_encoder.fit_transform(Comp_Data["Urban"])
Comp_Data["US"] = label_encoder.fit_transform(Comp_Data["US"])
Comp_Data["Sales1"] = label_encoder.fit_transform(Comp_Data["Sales1"])


# In[24]:


Comp_Data


# In[26]:


x=Comp_Data.iloc[:,1:11]
x


# In[28]:


y=Comp_Data["Sales1"]
y


# In[29]:


Comp_Data.Sales1.value_counts()


# In[31]:


colnames=list(Comp_Data.columns)
colnames


# ## Split Data into Train and Test 

# In[32]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)


# In[34]:


model = DecisionTreeClassifier(criterion = 'entropy')


# In[35]:


model.fit(x_train,y_train)


# # Build Tree Model

# In[36]:


tree.plot_tree(model);


# In[38]:


fn=['CompPrice','Income','Advertising','Population','Price','ShelveLoc','Age','Education','Urban','US']
cn=['Not High Sales', 'High Sales']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (8,7), dpi=600)
tree.plot_tree(model,feature_names = fn, class_names=cn,filled = True);


# In[39]:


preds=model.predict(x_test)
pd.Series(preds).value_counts()


# In[40]:


pd.Series(y_test).value_counts()


# In[41]:


pd.crosstab(y_test,preds)


# In[42]:


np.mean(preds==y_test)


# # Building Decision Tree Classifier (CART) using Gini Criteria

# In[43]:


model_gini = DecisionTreeClassifier(criterion='gini')


# In[44]:


model_gini.fit(x_train, y_train)


# In[45]:


#Prediction and computing the accuracy
pred=model.predict(x_test)
np.mean(preds==y_test)


# # Decision Tree Regression

# In[46]:


array=Comp_Data.values


# In[48]:


X=array[:,1:11]
X


# In[50]:


y=array[:,-1]
y


# In[51]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20)


# In[52]:


from sklearn.tree import DecisionTreeRegressor
model1=DecisionTreeRegressor()


# In[53]:


model1.fit(X_train, y_train)


# In[54]:


#Find the accuracy
model1.score(X_test,y_test)


# In[ ]:


#This Dataset is not Good for Decision Tree Regrssion


# In[ ]:





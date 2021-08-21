#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Load Dataset

# In[4]:


ps = pd.read_csv("PlasticSales.csv")
ps.head(10)


# In[5]:


ps.shape


# In[6]:


ps.dtypes


# In[7]:


ps.info()


# In[9]:


ps.describe()


# # Visualization

# In[10]:


plt.figure(figsize=(24,5))
ps.Sales.plot()


# In[11]:


ps["Date"] = pd.to_datetime(ps.Month,format="%b-%y")
#look for c standard format codes

# Extracting Day, weekday name, month name, year from the Date column using 
# Date functions from pandas 

ps["month"] = ps.Date.dt.strftime("%b") # month extraction
ps["year"] = ps.Date.dt.strftime("%y") # year extraction

#ps["Day"] = ps.Date.dt.strftime("%d") # Day extraction
#ps["wkday"] = ps.Date.dt.strftime("%A") # weekday extraction


# In[12]:


plt.figure(figsize=(12,8))
heatmap_y_month = pd.pivot_table(data=ps,values="Sales",index="year",columns="month",aggfunc="mean",fill_value=0)
sns.heatmap(heatmap_y_month,annot=True,fmt="g") #fmt is format of the grid values


# In[13]:


# Boxplot for ever
plt.figure(figsize=(8,6))
plt.subplot(211)
sns.boxplot(x="month",y="Sales",data=ps)
plt.subplot(212)
sns.boxplot(x="year",y="Sales",data=ps)


# In[14]:


month =['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'] 
#import numpy as np
p = ps["Month"][0]
p[0:3]
ps['months']= 0

for i in range(60):
    p = ps["Month"][i]
    ps['months'][i]= p[0:3]
    
month_dummies = pd.DataFrame(pd.get_dummies(ps['months']))
ps1 = pd.concat([ps.Sales,month_dummies],axis = 1)

ps1["t"] = np.arange(1,61)

ps1["t_square"] = ps1["t"]*ps1["t"]
ps1.columns
ps1["log_Sales"] = np.log(ps1["Sales"])
ps1.rename(columns={"Sales ": 'Sales'}, inplace=True)
ps1.Sales.plot()


# In[15]:


ps1


# In[16]:


plt.figure(figsize=(12,3))
sns.lineplot(x="year",y="Sales",data=ps)


# # Splitting the Data

# In[17]:


Train = ps1.head(35)


# In[18]:


Test = ps1.iloc[35:48,:]


# In[19]:


predict_data = ps1.tail(12)


# In[20]:


ps2= ps1.iloc[0:48,:]


# In[21]:


Train


# In[22]:


Test


# In[23]:


predict_data


# # Build Model & RMSE Value

# ### Linear Model
# 

# In[24]:


import statsmodels.formula.api as smf 
linear_model = smf.ols('Sales~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_linear))**2))
rmse_linear


# # Exponential
# 
# 

# In[25]:


Exp = smf.ols('log_Sales~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp


# ### Quadratic 
# 
# 

# In[26]:


Quad = smf.ols('Sales~t+t_square',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_square"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_Quad))**2))
rmse_Quad


# ### Additive seasonality 
# 
# 

# In[27]:


add_sea = smf.ols('Sales~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea))**2))
rmse_add_sea


# ### Additive Seasonality Quadratic 
# 
# 

# In[28]:


add_sea_Quad = smf.ols('Sales~t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_square']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad


# ### Multiplicative Seasonality
# 
# 

# In[29]:


Mul_sea = smf.ols('log_Sales~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea


# ### Multiplicative Additive Seasonality 
# 
# 

# In[30]:


Mul_Add_sea = smf.ols('log_Sales~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea


# ### Compare the results 
# 
# 

# In[31]:


data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse.sort_values(['RMSE_Values'])


# # Predict the New Model

# In[32]:


predict_data


# ### Build the model on entire data set

# In[33]:


model_full = smf.ols('log_Sales~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=ps2).fit()


# In[35]:


pred_new  = pd.Series(Mul_Add_sea.predict(predict_data))
pred_new


# In[36]:


predict_data["forecasted_Sales"] = pd.DataFrame(pred_new)


# In[37]:


predict_data


# In[ ]:





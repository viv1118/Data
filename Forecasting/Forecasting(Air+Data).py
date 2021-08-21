#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Load Dataset 

# In[3]:


Air=pd.read_excel("Airlines+Data.xlsx")
Air.head(10)


# In[5]:


Air.shape


# In[6]:


Air.dtypes


# In[7]:


Air.info()


# In[9]:


Air.describe()


# # Visualization

# In[10]:


plt.figure(figsize=(24,5))
Air.Passengers.plot()


# In[11]:


Air["Date"] = pd.to_datetime(Air.Month,format="%b-%y")
#look for c standard format codes

# Extracting Day, weekday name, month name, year from the Date column using 
# Date functions from pandas 

Air["month"] = Air.Date.dt.strftime("%b") # month extraction
Air["year"] = Air.Date.dt.strftime("%y") # year extraction

#Air["Day"] = Air.Date.dt.strftime("%d") # Day extraction
#Air["wkday"] = Air.Date.dt.strftime("%A") # weekday extractio


# In[12]:


Air


# In[14]:


plt.figure(figsize=(12,8))
heatmap_y_month = pd.pivot_table(data=Air,values="Passengers",index="year",columns="month",aggfunc="mean",fill_value=0)
sns.heatmap(heatmap_y_month,annot=True,fmt="g") #fmt is format of the grid values


# In[15]:


# Boxplot for ever
plt.figure(figsize=(8,6))
plt.subplot(211)
sns.boxplot(x="month",y="Passengers",data=Air)
plt.subplot(212)
sns.boxplot(x="year",y="Passengers",data=Air)


# In[16]:


month =['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'] 
#import numpy as np
A = Air["month"][0]
A[0:3]
Air['Month']= 0

for i in range(96):
    A = Air["month"][i]
    Air['month'][i]= A[0:3]
    
month_dummies = pd.DataFrame(pd.get_dummies(Air['month']))
Air1 = pd.concat([Air.Passengers,month_dummies],axis = 1)

Air1["t"] = np.arange(1,97)

Air1["t_square"] = Air1["t"]*Air1["t"]
Air1.columns
Air1["log_Passengers"] = np.log(Air1["Passengers"])
Air1.rename(columns={"Passengers ": 'Passengers'}, inplace=True)
Air1.Passengers.plot()


# In[18]:


Air1


# In[19]:


plt.figure(figsize=(12,3))
sns.lineplot(x="year",y="Passengers",data=Air)


# In[23]:


Air.Passengers.plot(label="org")
for i in range(2,10,2):
    Air["Passengers"].rolling(i).mean().plot(label=str(i))
plt.legend(loc=3)


# In[24]:


Air.Passengers.plot()


# # Splitting data

# In[26]:


Train = Air1.head(72)


# In[27]:


Test = Air1.iloc[72:84,:]


# In[28]:


predict_data = Air1.tail(12)


# In[30]:


Air2= Air1.iloc[0:84,:]


# In[31]:


Train


# In[32]:


Test


# In[33]:


predict_data


# # Build Model & RMSE Value

# ### Linear Model

# In[34]:


import statsmodels.formula.api as smf 
linear_model = smf.ols('Passengers~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_linear))**2))
rmse_linear


# ### Exponential
# 
# 

# In[35]:


Exp = smf.ols('log_Passengers~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp


# ### Quadratic 
# 
# 

# In[36]:


Quad = smf.ols('Passengers~t+t_square',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_square"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_Quad))**2))
rmse_Quad


# ### Additive seasonality 
# 
# 

# In[37]:


add_sea = smf.ols('Passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea))**2))
rmse_add_sea


# ### Additive Seasonality Quadratic 
# 
# 

# In[38]:


add_sea_Quad = smf.ols('Passengers~t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_square']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad


# ### Multiplicative Seasonality
# 
# 

# In[39]:


Mul_sea = smf.ols('log_Passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea


# ### Multiplicative Additive Seasonality 
# 
# 

# In[40]:


Mul_Add_sea = smf.ols('log_Passengers~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea


# ### Compare the results 
# 
# 

# In[45]:


data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse.sort_values(['RMSE_Values'])


# # Predict for new time model

# In[46]:


predict_data


# ### Build the model on entire data set
# 

# In[47]:


model_full = smf.ols('log_Passengers~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Air2).fit()


# In[49]:


pred_new  = pd.Series(Mul_Add_sea.predict(predict_data))
pred_new


# In[50]:


predict_data["forecasted_Passengers"] = pd.DataFrame(pred_new)


# In[51]:


predict_data


# In[ ]:





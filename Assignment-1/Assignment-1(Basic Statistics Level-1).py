#!/usr/bin/env python
# coding: utf-8

# # Assignment-1-Q7 (Basic Statistics Level-1)

# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


data1=pd.read_csv("Q7.csv")
data1


# In[10]:


# mean
data1.mean()


# In[11]:


data1.median()


# In[13]:


# Mode
data1.Points.mode()


# In[14]:


data1.Score.mode()


# In[15]:


data1.Weigh.mode()


# In[16]:


# Variance
data1.var()


# In[17]:


# Satndard Deviation
data1.std()


# In[18]:


data1.describe()


# In[20]:


#Range
Points_Range=data1.Points.max()-data1.Points.min()
Points_Range


# In[22]:


Score_Range=data1.Score.max()-data1.Score.min()
Score_Range


# In[24]:


Weigh_Range=data1.Weigh.max()-data1.Weigh.min()
Weigh_Range


# # Visualization

# In[25]:


f,ax=plt.subplots(figsize=(15,5))
plt.subplot(1,3,1)
plt.boxplot(data1.Points)
plt.title('Points')
plt.show()


# In[26]:


plt.subplot(1,3,2)
plt.boxplot(data1.Score)
plt.title('Score')
plt.show()


# In[27]:


plt.subplot(1,3,3)
plt.boxplot(data1.Weigh)
plt.title('Weigh')
plt.show()


# In[28]:


# Inferences: a) For Points dataset: 1) The data is concentrated aroound Median 
#2) There are no outliars
#3) The distribution is Right skewed 
#b) For Score dataset: 1) The data is concentrated around Median 
#2) There are 3 Outliars: 5.250, 5.424, 5.345 
#3) The distribution is Left skewed
#c) For Weigh dataset: 1) The data is concentrated around Median 
#2) There is 1 Outliar: 22.90 
#3) The distribution is Left skewed


# In[ ]:





# # Assignment-1-Q9_a (Basic Statistics Level-1)

# In[29]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[32]:


data2=pd.read_csv("Q9_a.csv")
data2


# In[33]:


# Skewness
data2.skew()


# In[34]:


# Skewness Inference: 1. Speed distribution is left skewed (negative skewness) 
#2. Distance distributin is right skewed (positive skewness)


# In[35]:


# Kurtosis
data2.kurt()


# In[36]:


#Kurtosis Inference: 1. Speed distribution is platykurtic (negative kurtosis i.e. flatter than normal distribution)
#2. Distance distributin is leptokurtic (positive kurtosis i.e. peaked than noramal distribution)


# In[ ]:





# # Assignment-1-Q9_b (Basic Statistics Level-1)

# In[37]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[39]:


data3=pd.read_csv("Q9_b.csv")
data3


# In[45]:


data4=data3.iloc[:,1:]
data4


# In[46]:


# Skewness
data4.skew()


# In[47]:


## Skewness Inference: 1. WT distribution is left skewed (negative skewness) 
#2. SP distributin is right skewed (positive skewness)


# In[48]:


# Kurtosis
data4.kurt()


# In[49]:


#Kurtosis Inference:SP & WT distribution both are leptokurtic (positive kurtosis i.e. peaked than noramal distribution)


# In[ ]:





# # Assignment-1-Q11 (Basic Statistics Level-1)

# In[50]:


import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm


# In[51]:


# Given,Sample mean = 200 Sample SD = 30 n = 2000 


# In[52]:


# Avg. weight of Adult in Mexico with 94% CI
stats.norm.interval(0.94,200,30/(2000**0.5))


# In[53]:


# Avg. weight of Adult in Mexico with 96% CI
stats.norm.interval(0.96,200,30/(2000**0.5))


# In[54]:


# Avg. weight of Adult in Mexico with 98% CI
stats.norm.interval(0.98,200,30/(2000**0.5))


# In[ ]:





# # Assignment-1-Q12 (Basic Statistics Level-1)

# In[55]:


import numpy as np
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[57]:


x=pd.Series([34,36,36,38,38,39,39,40,40,41,41,41,41,42,42,45,49,56])
x


# In[58]:


sns.boxplot(x)


# In[59]:


#Mean 
x.mean()


# In[60]:


#Median
x.median()


# In[61]:


#Variance
x.var()


# In[62]:


#Standard Deviation
x.std()


# In[63]:


plt.boxplot(x)


# In[65]:


#2).Inference:\n",
   # "1. There are 2 Outliars in Student's marks: 49 and 56"


# In[ ]:





# # Assignment-1-Q12 (Basic Statistics Level-1)

# In[66]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[68]:


cars=pd.read_csv('Cars.csv')
cars


# In[69]:


cars.describe()


# In[75]:


name=['HP', 'MPG', 'VOL', 'SP', 'WT'] 
name


# # Visualization

# In[82]:


sns.boxplot(cars.MPG)


# In[83]:


sns.boxplot(cars.HP)


# In[84]:


sns.boxplot(cars.VOL)


# In[85]:


sns.boxplot(cars.SP)


# In[86]:


sns.boxplot(cars.WT)


# In[91]:


cars.MPG.mean()


# In[92]:


cars.MPG.std()


# In[95]:


# P(MPG>38)
stats.norm.cdf(38,cars.MPG.mean(),cars.MPG.std())


# In[96]:


# P(MPG<40)
stats.norm.cdf(40,cars.MPG.mean(),cars.MPG.std())


# In[98]:


# P (20<MPG<50)
stats.norm.cdf(50,cars.MPG.mean(),cars.MPG.std())-stats.norm.cdf(20,cars.MPG.mean(),cars.MPG.std())


# In[ ]:





# # Assignment-1-Q21_a (Basic Statistics Level-1)

# In[99]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[101]:


cars=pd.read_csv('Cars.csv')
cars


# In[102]:


sns.distplot(cars.MPG, label='Cars-MPG')
plt.xlabel('MPG')
plt.ylabel('Density')
plt.legend();


# In[106]:


cars.MPG.mean()


# In[107]:


cars.MPG.median()


# In[108]:


#Inference: MPG of Cars does follow normal distribution approximately (as mean and median are approx. same)


# In[ ]:





# # Assignment-1-Q21_b (Basic Statistics Level-1)

# In[110]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[112]:


wcat=pd.read_csv('wc-at.csv')
wcat


# In[113]:


# plotting distribution for Waist Tissue (WT)
sns.distplot(wcat.Waist)
plt.ylabel('density')
plt.show()


# In[114]:


# plotting distribution for Adipose Tissue (AT)
sns.distplot(wcat.AT)
plt.ylabel('density')
plt.show()


# In[115]:


# WC
wcat.Waist.mean()  


# In[116]:


wcat.Waist.median()


# In[117]:


# AT
wcat.AT.mean() 


# In[118]:


wcat.AT.median()


# In[119]:


#Inference: Both the Adipose Tissue (AT) and Waist Circumference(Waist) data set do not follow the normal distribution approximately (as mean and median of both the data are approximately different)


# In[ ]:





# # Assignment-1-Q22 (Basic Statistics Level-1)

# In[120]:


from scipy import stats
from scipy.stats import norm


# In[121]:


# Z-score of 90% confidence interval 
stats.norm.ppf(0.95)


# In[123]:


# Z-score of 94% confidence interval
stats.norm.ppf(0.94)


# In[126]:


# Z-score of 60% confidence interval
stats.norm.ppf(0.60)


# In[ ]:





# # Assignment-1-Q23 (Basic Statistics Level-1)

# In[127]:


from scipy import stats
from scipy.stats import norm


# In[128]:


n=25
df=n-1
df


# In[129]:


#t scores of 95% confidence interval for sample size of 25
stats.t.ppf(0.95,24)


# In[130]:


#t scores of 96% confidence interval for sample size of 25
stats.t.ppf(0.96,24)


# In[131]:


#t scores of 99% confidence interval for sample size of 25
stats.t.ppf(0.99,24)


# In[ ]:





# # Assignment-1-Q24 (Basic Statistics Level-1)

# In[133]:


from scipy import stats
from scipy.stats import norm


# In[136]:


#P_mean(Pop mean) =270 days S_mean(sample mean) = 260 days Sample SD = 90, days Sample n = 18 bulbs


# In[138]:


n = 18
df = n-1 
df


# In[139]:


# Assume Null Hypothesis is: Ho = Avg life of Bulb >= 260 days
# Alternate Hypothesis is: Ha = Avg life of Bulb < 260 days


# In[141]:


# find t-scores at x=260; t=(s_mean-P_mean)/(s_SD/sqrt(n))
t=(260-270)/(90/18**0.5)
t


# In[143]:


# p_value=1-stats.t.cdf(abs(t_scores),df=n-1)... Using cdf function
p_value=1-stats.t.cdf(abs(-0.4714),df=17)
p_value


# In[145]:


#  OR p_value=stats.t.sf(abs(t_score),df=n-1)... Using sf function
p_value=stats.t.sf(abs(-0.4714),df=17)
p_value


# In[146]:


#Probability that 18 randomly selected bulbs would have an average life of no more than 260 days is 32.17%Assuming significance value α = 0.05 (Standard Value)(If p_value < α ; Reject Ho and accept Ha or vice-versa)Thus, as p-value > α ; Accept Ho i.e. The CEO claims are false and the avg life of bulb > 260 days


# In[ ]:





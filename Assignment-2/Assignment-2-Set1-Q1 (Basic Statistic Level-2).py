#!/usr/bin/env python
# coding: utf-8

# # Assignment-2-Set1-Q1 (Basic Statistic Level-2)

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


x=pd.Series([24.23,25.53,25.41,24.14,29.62,28.25,25.81,24.39,40.26,32.95,91.36,25.99,39.42,26.71,35.00])
x


# In[5]:


name=['Allied Signal','Bankers Trust','General Mills','ITT Industries','J.P.Morgan & Co.','Lehman Brothers',
      'Marriott','MCI','Merrill Lynch','Microsoft','Morgan Stanley','Sun Microsystems','Travelers','US Airways',
      'Warner-Lambert']
name


# In[6]:


plt.figure(figsize=(6,8))
plt.pie(x,labels=name,autopct='%1.0f%%')
plt.show()


# In[7]:


sns.boxplot(x)


# In[8]:


x.mean()


# In[9]:


x.var()


# In[10]:


x.std()


# In[ ]:





# # Assignment-2-Set1-Q5-(iv) (Basic Statistic Level-2)

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


y =pd.Series([0.1,0.1,0.2,0.2,0.3,0.1])
y


# In[4]:


sns.boxplot(y)


# In[5]:


y.std()


# In[ ]:





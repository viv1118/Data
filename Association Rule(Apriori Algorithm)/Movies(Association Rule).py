#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder


# # Load Dataset

# In[2]:


movies = pd.read_csv("my_movies.csv")
movies


# ## Pre-Processing

# In[3]:


df=pd.get_dummies(movies)
df


# In[4]:


df.describe()


# # Apriori Algorithm

# In[6]:


frequent_itemsets = apriori(df,min_support=0.5,use_colnames=True)
frequent_itemsets


# In[8]:


rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.7)
rules


# In[9]:


len(rules)


# In[11]:


rules1 = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.9)
rules1


# In[12]:


rules.sort_values('lift',ascending = False)


# In[13]:


rules[rules.lift>1]


# # Visualization

# In[16]:


import matplotlib.pyplot as plt
plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()


# In[17]:


plt.scatter(rules['support'], rules['lift'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('lift')
plt.title('Support vs Lift')
plt.show()


# In[18]:


import numpy as np
fit = np.polyfit(rules['lift'], rules['confidence'], 1)
fit_fn = np.poly1d(fit)
plt.plot(rules['lift'], rules['confidence'], 'yo', rules['lift'], 
 fit_fn(rules['lift']))


# In[ ]:





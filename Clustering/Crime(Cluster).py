#!/usr/bin/env python
# coding: utf-8

# # import Libraries

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from numpy import random, float, array
import numpy as np
import seaborn as sns


# # Load Data

# In[3]:


crime=pd.read_csv("crime_data.csv")
crime.head(10)


# In[4]:


crime.shape


# In[5]:


crime.info()


# In[6]:


#Normalization Function
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)


# In[8]:


# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(crime.iloc[:,1:])
df_norm.describe()


# In[10]:


from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch # for creating dendrogram


# In[12]:


z = linkage(df_norm, method="complete",metric="euclidean")
z


# In[13]:


crime.corr()


# # Visualization

# In[14]:


plt.figure(figsize=(15, 5))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Features')
plt.ylabel('Crime')
sch.dendrogram(z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()


# In[16]:


###### screw plot or elbow curve ############
k = list(range(2,15))
k


# In[17]:


from sklearn.cluster import	KMeans
from scipy.spatial.distance import cdist 
import numpy as np


# In[23]:


TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(df_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))


# In[24]:


# Scree plot 
plt.figure(figsize=(16,6))
plt.plot(k,TWSS,'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)


# # The elbow appear to be smoothening out after four clusters indicating that the optimal number of clusters is 4.

# In[27]:


# Selecting 4 clusters from the above scree plot which is the optimum number of clusters 
model=KMeans(n_clusters=4) 
model.fit(df_norm)


# In[28]:


KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
    n_clusters=4, n_init=10, n_jobs=None, precompute_distances='auto',
    random_state=None, tol=0.0001, verbose=0)


# In[29]:


model.labels_ # getting the labels of clusters assigned to each row 


# In[30]:


model.cluster_centers_


# In[31]:


X = crime[['Murder', 'Assault', 'Rape', 'UrbanPop']]
clusters = KMeans(4)  # 4 clusters!
clusters.fit( X )
clusters.cluster_centers_
clusters.labels_
crime['Crime_clusters'] = clusters.labels_
crime.head()
crime.sort_values(by=['Crime_clusters'],ascending = True)
X.head()


# In[33]:


stats =crime.sort_values("Murder", ascending=True)
stats


# In[34]:


# Plot between pairs Murder~Assault
sns.lmplot( 'Murder','Assault',  data=crime,
        hue = 'Crime_clusters',
        fit_reg=False, size = 6 );


# In[35]:


# Plot between pairs Murder~Rape
sns.lmplot( 'Murder','Rape',  data=crime,
        hue = 'Crime_clusters',
        fit_reg=False, size = 6 );


# In[36]:


# Plot between pairs Assault~Rape
sns.lmplot( 'Assault','Rape',  data=crime,
        hue = 'Crime_clusters',
        fit_reg=False, size = 6 );


# # All dots are states of US and different colors are one cluster showing clustering for the crime data.

# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[2]:


import pandas as pd 
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import scale


# # Load Dataset

# In[4]:


wine = pd.read_csv("wine.csv")
wine.head(10)


# In[5]:


wine.shape


# In[6]:


wine.dtypes


# In[7]:


wine.info()


# In[8]:


wine.describe()


# In[9]:


wine.data = wine.iloc[:,1:]


# In[10]:


wine.data.head()


# In[11]:


wine_normal = scale(wine.data)


# In[12]:


wine_normal


# In[13]:


pca = PCA()
pca_values = pca.fit_transform(wine_normal)


# In[14]:


pca_values


# In[23]:


pca = PCA(n_components = 6)
pca.fit(wine)


# In[17]:


#pca_values = pca.fit_transform(uni_normal)


# # Visualization

# In[18]:


sns.pairplot(pd.DataFrame(pca_values))


# # PCA

# In[24]:


var = pca.explained_variance_ratio_.cumsum()
var


# In[25]:


# Cumulative variance 
var1 = np.cumsum(np.round(var,decimals = 4)*100)
var1


# In[26]:


pca.components_


# In[27]:


plt.plot(var1,color="red")


# In[28]:


x = pca_values[:,0:1]
y = pca_values[:,1:2]
#z = pca_values[:2:3]
plt.scatter(x,y)


# In[29]:


finalDf = pd.concat([pd.DataFrame(pca_values[:,0:7],columns=['pc1','pc2','pc3','pc4','pc5','pc6','pc7']), wine[['Type']]], axis = 1)


# In[30]:


finalDf


# In[31]:


sns.scatterplot(data=finalDf,x='pc1',y='pc2',hue='Type')


# In[32]:


pcavalues=pd.DataFrame(pca_values[:,:7],columns=['pc1','pc2','pc3','pc4','pc5','pc6','pc7'])


# In[33]:


pcavalues


# # Hierarichal Clustering

# In[34]:


import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(pcavalues, method='complete'))


# In[35]:


#No infrences can be derived from the dendrogram.. We can go for Kmean Clustering for large data sets.


# # Kmeans

# In[36]:


from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


# In[37]:


k = list(range(2,8))
k
TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(pcavalues)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(pcavalues.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,pcavalues.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))
    
TWSS


# In[38]:


#Elbow Chart
plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)


# In[40]:


kmeans_clust=KMeans(n_clusters=5)
kmeans_clust.fit(pcavalues)
Clusters=pd.DataFrame(kmeans_clust.labels_,columns=['Clusters'])
Clusters


# In[41]:


wine['h_clusterid'] = pd.DataFrame(Clusters)


# In[42]:


wine


# ### Grouping Data for predictions further 
# 
# 

# In[43]:


result=wine.iloc[:,1:].groupby(wine.h_clusterid).mean()


# In[44]:


result


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# # Load Dataset

# In[3]:


xls = pd.ExcelFile('EastWestAirlines.xlsx')
air = pd.read_excel(xls,'data',sep=';')


# In[4]:


air.head()


# In[6]:


air.shape


# In[7]:


air.info()


# In[9]:


# Dataset column renaming

air.columns = ['ID', 'Balance', 'Qual_miles', 'cc1_miles', 'cc2_miles', 'cc3_miles','Bonus_miles', 'Bonus_trans', 
                 'Flight_miles_12mo', 'Flight_trans_12','Days_since_enroll', 'Award'] 


# In[10]:


air.info()


# In[11]:


air.head(10)


# In[12]:


# writing loop to check datatype other than integer, if found any that will be replaced by Nan.

def column_preprocessor(df):
    count = 0
    for row in df:
        try:
            if type(row) != int:
                df.loc[count] = np.nan
        except:
            pass
        count +=1


# In[13]:


column_preprocessor(air[air.columns])


# In[14]:


air.isna().any().sum()


# # EDA

# In[15]:


air.describe().transpose()


# In[16]:


air.head(2)


# In[17]:


air['Award'].value_counts().plot(kind='pie', autopct='%2.0f%%', fontsize='18', 
                                        colors = ['#F11A05','#43E206'], shadow =True)
plt.show()


# In[18]:


fig, ax =plt.subplots(figsize=(40,8))
ax = sns.lineplot(x= 'Days_since_enroll', y='Balance',data = air)


# In[19]:


sns.heatmap(air.corr()>0.85, annot=True)


# # Data Preprocessing

# In[22]:


air1 =  air.drop(['ID','Award'], axis=1)
air1.head(2)


# # Standardization

# In[24]:


from sklearn.preprocessing import StandardScaler
std_df = StandardScaler().fit_transform(air1)
std_df.shape


# # MinMaxScaler

# In[26]:


from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()
minmax_df = minmax.fit_transform(air1)
minmax_df.shape


# # PCA

# In[27]:


from sklearn.decomposition import PCA
pca_std = PCA(random_state=10, n_components=0.95)
pca_std_df= pca_std.fit_transform(std_df)


# In[28]:


print(pca_std.singular_values_)


# In[29]:


print(pca_std.explained_variance_ratio_*100)


# In[30]:


cum_variance = np.cumsum(pca_std.explained_variance_ratio_*100)
cum_variance


# In[31]:


from sklearn.decomposition import PCA
pca_minmax =  PCA(random_state=10, n_components=0.95)
pca_minmax_df = pca_minmax.fit_transform(minmax_df)


# In[32]:


print(pca_minmax.singular_values_)


# In[33]:


print(pca_minmax.explained_variance_ratio_*100)


# # K-Means Clstering

# In[35]:


#import the KElbowVisualizer Method
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer


# In[36]:


model1 = KMeans(random_state=0,n_jobs=-1)


# In[37]:


model2 = KMeans(random_state=10, n_jobs=-1, max_iter=500, n_init=20)


# In[38]:


visualizer1 = KElbowVisualizer(model1, k=(2,10), metric='silhouette', timings=False)


# In[39]:


visualizer2 = KElbowVisualizer(model2, k=(2,10), metric='silhouette', timings=False)


# In[40]:


print('model1')
visualizer1.fit(pca_std_df)    
visualizer1.poof()
plt.show()


# In[41]:


print('model2')
visualizer2.fit(pca_std_df)    
visualizer2.poof()
plt.show()


# In[42]:


from sklearn.metrics import silhouette_score
list1= [2,3,4,5,6,7,8,9]  # always start number from 2.
for n_clusters in list1:
    clusterer1 = KMeans(n_clusters=n_clusters, random_state=0,n_jobs=-1)
    cluster_labels1 = clusterer1.fit_predict(pca_std_df)
    sil_score1= silhouette_score(pca_std_df, cluster_labels1)
    print("For n_clusters =", n_clusters,"The average silhouette_score is :", sil_score1)


# In[43]:


from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer


# In[44]:


model3 = KMeans(random_state=0,n_jobs=-1)


# In[45]:


model4 = KMeans(random_state=10, n_jobs=-1, max_iter=500, n_init=20)


# In[46]:


visualizer3 = KElbowVisualizer(model3, k=(2,10), metric='silhouette', timings=False)


# In[47]:


visualizer4 = KElbowVisualizer(model4, k=(2,10), metric='silhouette', timings=False)


# In[48]:


print('model3')
visualizer3.fit(pca_minmax_df)    
visualizer3.poof()
plt.show()


# In[49]:


print('model4')
visualizer4.fit(pca_minmax_df)    
visualizer4.poof()
plt.show()


# In[50]:


from sklearn.metrics import silhouette_score

list1= [2,3,4,5,6,7,8,9]  # always start number from 2.

for n_clusters in list1:
    clusterer2 = KMeans(n_clusters=n_clusters, random_state=0,n_jobs=-1)
    cluster_labels2 = clusterer1.fit_predict(pca_minmax_df)
    sil_score2= silhouette_score(pca_std_df, cluster_labels2)
    print("For n_clusters =", n_clusters,"The average silhouette_score is :", sil_score2)


# # K-Means Clustering Algorithm

# In[52]:


model1 = KMeans(n_clusters=6, random_state=0,n_jobs=-1)
y_predict1 = model1.fit_predict(pca_std_df)
y_predict1.shape


# In[53]:


y_predict1


# In[54]:


model1.labels_


# In[55]:


model1.cluster_centers_


# In[56]:


model1.inertia_


# In[57]:


model1.score(pca_std_df) 


# In[58]:


model1.get_params()


# In[67]:


from yellowbrick.cluster import SilhouetteVisualizer
fig,(ax1,ax2) = plt.subplots(1,2,sharey=False)
fig.set_size_inches(15,6)
sil_visualizer1 = SilhouetteVisualizer(model1,ax= ax1, colors=['#922B21','#5B2C6F','#1B4F72','#32a84a','#a83232','#323aa8'])
sil_visualizer1.fit(pca_std_df)
import matplotlib.cm as cm
colors1 = cm.nipy_spectral(model1.labels_.astype(float) / 6) # 6 is number of clusters
ax2.scatter(pca_std_df[:, 0], pca_std_df[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors1, edgecolor='k')
centers1 = model1.cluster_centers_
ax2.scatter(centers1[:, 0], centers1[:, 1], marker='o',c="white", alpha=1, s=200, edgecolor='k')
for i, c in enumerate(centers1):
    ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,s=50, edgecolor='k')
ax2.set_title(label ="The visualization of the clustered data.")
ax2.set_xlabel("Feature space for the 1st feature")
ax2.set_ylabel("Feature space for the 2nd feature")
plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % 6),fontsize=14, fontweight='bold')
sil_visualizer1.show()
plt.show()


# In[68]:


centers1 = model1.cluster_centers_
centers1


# # Putting Cluster lables into original dataset And analysis of the same

# In[70]:


model1_cluster = pd.DataFrame(model1.labels_.copy(), columns=['Kmeans_Clustering'])
model1_cluster


# In[73]:


Kmeans_df = pd.concat([air.copy(), model1_cluster], axis=1)
Kmeans_df.head()


# In[74]:


fig, ax = plt.subplots(figsize=(10, 6))
Kmeans_df.groupby(['Kmeans_Clustering']).count()['ID'].plot(kind='bar')
plt.ylabel('ID Counts')
plt.title('Kmeans Clustering (pca_std_df)',fontsize='large',fontweight='bold')
ax.set_xlabel('Clusters', fontsize='large', fontweight='bold')
ax.set_ylabel('ID counts', fontsize='large', fontweight='bold')
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.show()


# # Hierarchical Clustering Algorithm

# # 1. using By Dendrogram

# In[100]:


import scipy.cluster.hierarchy as shc
for methods in ['single','complete','average','weighted','centroid','median','ward']:
    plt.figure(figsize =(20, 6))
    dict={'fontsize':24,'fontweight':16,'color':'blue'}
    plt.title('Visualising the data, Methods-{}'.format(methods),fontdict = dict) 
    Dendrogram1 = shc.dendrogram(shc.linkage(pca_std_df, method = methods,optimal_ordering=False))  


# # Method 2: Silhouette Score method
#     
#  
# 

# In[101]:


from sklearn.cluster import AgglomerativeClustering
n_clusters = [2,3,4,5,6,7,8]  
for n_clusters in n_clusters:
    for linkages in ["ward", "complete", "average", "single"]:
        hie_cluster1 = AgglomerativeClustering(n_clusters=n_clusters,linkage=linkages) # bydefault it takes linkage 'ward'
        hie_labels1 = hie_cluster1.fit_predict(pca_std_df)
        silhouette_score1 = silhouette_score(pca_std_df, hie_labels1)
        print("For n_clusters =", n_clusters,"The average silhouette_score with linkage-",linkages, ':',silhouette_score1)
    print()


# In[102]:


from sklearn.cluster import AgglomerativeClustering
n_clusters = [2,3,4,5,6,7,8]  
for n_clusters in n_clusters:
    for linkages in ["ward", "complete", "average", "single"]:
        hie_cluster2 = AgglomerativeClustering(n_clusters=n_clusters,linkage=linkages) # bydefault it takes linkage 'ward'
        hie_labels2 = hie_cluster2.fit_predict(pca_minmax_df)
        silhouette_score2 = silhouette_score(pca_minmax_df, hie_labels2)
        print("For n_clusters =", n_clusters,"The average silhouette_score with linkage-",linkages, ':',silhouette_score2)
    print()


# In[103]:


agg_clustering = AgglomerativeClustering(n_clusters=5, linkage='average')
y_pred_hie = agg_clustering.fit_predict(pca_std_df)
print(y_pred_hie.shape)


# In[104]:


y_pred_hie


# In[105]:


agg_clustering.n_clusters_


# In[106]:


agg_clustering.labels_


# In[107]:


agg_clustering.n_leaves_


# In[108]:


agg_clustering.n_connected_components_


# In[109]:


agg_clustering.children_


# In[110]:


(silhouette_score(pca_std_df, agg_clustering.labels_)*100).round(3)


# In[111]:


import scipy.cluster.hierarchy as shc
for methods in ['average']: 
    plt.figure(figsize =(20, 6)) 
    dict = {'fontsize':24,'fontweight' :16, 'color' : 'blue'}
    plt.title('Visualising the data, Method- {}'.format(methods),fontdict = dict) 
    Dendrogram2 = shc.dendrogram(shc.linkage(pca_std_df, method = methods,optimal_ordering=False))


# # Putting Cluster lables into original dataset And analysis of the same

# In[113]:


hie_cluster = pd.DataFrame(agg_clustering.labels_.copy(), columns=['Hie_Clustering'])
hie_cluster


# In[115]:


hie_df = pd.concat([air.copy(), hie_cluster], axis=1)
hie_df .head()


# In[119]:


fig, ax = plt.subplots(figsize=(10, 6))
hie_df.groupby(['Hie_Clustering']).count()['ID'].plot(kind='bar')
plt.ylabel('ID Counts')
plt.title('Hierarchical Clustering (pca_std_df)',fontsize='large',fontweight='bold')
ax.set_xlabel('Clusters', fontsize='large', fontweight='bold')
ax.set_ylabel('ID counts', fontsize='large', fontweight='bold')
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.show()


# # Conclusion Between Kmeans & Hierarchical

# In[117]:


Kmeans_df.groupby(['Kmeans_Clustering']).count()


# In[118]:


hie_df.groupby(['Hie_Clustering']).count()


# In[121]:


count_df = Kmeans_df.groupby(['Kmeans_Clustering']).count()
count_df


# In[122]:


count = count_df.xs('ID' ,axis = 1)
count.plot(kind='bar', title= 'Nuber Counts')
plt.show()


# In[123]:


cluster1 = pd.DataFrame(Kmeans_df.loc[Kmeans_df.Kmeans_Clustering==0].mean(),columns= ['Cluster1_avg'])
cluster2 = pd.DataFrame(Kmeans_df.loc[Kmeans_df.Kmeans_Clustering==1].mean(),columns= ['Cluster2_avg'])
cluster3 = pd.DataFrame(Kmeans_df.loc[Kmeans_df.Kmeans_Clustering==2].mean(),columns= ['Cluster3_avg'])
cluster4 = pd.DataFrame(Kmeans_df.loc[Kmeans_df.Kmeans_Clustering==3].mean(),columns= ['Cluster4_avg'])
cluster5 = pd.DataFrame(Kmeans_df.loc[Kmeans_df.Kmeans_Clustering==4].mean(),columns= ['Cluster5_avg'])


# In[124]:


avg_df = pd.concat([cluster1,cluster2,cluster3,cluster4,cluster5],axis=1)
avg_df


# In[125]:


for i , row in avg_df.iterrows():
    fig = plt.subplots(figsize=(8,6))
    j = avg_df.xs(i ,axis = 0)
    plt.title(i, fontsize=16, fontweight=20)
    j.plot(kind='bar',fontsize=14)
    plt.show()
    print()


# In[ ]:





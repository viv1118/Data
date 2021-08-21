#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


import pandas as pd
import numpy as np


# # Load Dataset

# In[3]:


book_df = pd.read_csv('book.csv',encoding='latin-1')
book_df


# In[4]:


book_df1=book_df.drop(['Unnamed: 0'], axis=1).rename(columns={"User.ID": "User_ID", "Book.Title": "Book_Title","Book.Rating": "Book_Rating" })


# In[5]:


book_df1


# # Preprocessing

# In[6]:


#number of unique users in the dataset
User_ID_unique=book_df1.User_ID.unique()


# In[7]:


User_ID_unique=pd.DataFrame(User_ID_unique)


# In[8]:


len(book_df1.Book_Title.unique())


# In[9]:


user_book_df = book_df1.pivot_table(index='User_ID',
                                 columns='Book_Title',
                                 values='Book_Rating')


# In[10]:


user_book_df


# # Build the Model

# In[11]:


user_book_df.index = book_df1.User_ID.unique()


# In[12]:


user_book_df


# In[13]:


#Impute those NaNs with 0 values
user_book_df.fillna(0, inplace=True)


# In[14]:


user_book_df


# In[15]:


#Calculating Cosine Similarity between Users
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine, correlation


# In[16]:


user_sim = 1 - pairwise_distances( user_book_df.values,metric='cosine')


# In[17]:


user_sim


# In[18]:


user_sim.shape


# In[19]:


#Store the results in a dataframe
user_sim_df = pd.DataFrame(user_sim)


# In[20]:


#Set the index and column names to user ids 
user_sim_df.index = book_df1.User_ID.unique()
user_sim_df.columns = book_df1.User_ID.unique()


# In[21]:


user_sim_df.iloc[0:5, 0:5]


# In[22]:


np.fill_diagonal(user_sim, 0)
user_sim_df.iloc[0:5, 0:5]


# In[23]:


#Most Similar Users
user_sim_df.idxmax(axis=1)


# In[24]:


book_df1[(book_df1['User_ID']==276729) | (book_df1['User_ID']==276726)]


# In[25]:


user_1=book_df1[book_df1['User_ID']==276729]


# In[26]:


user_2=book_df1[book_df1['User_ID']==276726]


# In[27]:


user_2.Book_Title


# In[28]:


user_1.Book_Title


# In[29]:


pd.merge(user_1,user_2,on='Book_Title',how='outer')


# In[ ]:





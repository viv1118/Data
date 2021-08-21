#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
import numpy


# In[2]:


import pandas as pd
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


# # Load Dataset

# In[4]:


dataset = pd.read_csv("forestfires.csv")
dataset.head(10)


# In[5]:


dataset.shape


# In[6]:


dataset.dtypes


# In[7]:


dataset.info()


# In[8]:


dataset.describe()


# ### Label Encoding

# In[9]:


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
dataset["month"] = label_encoder.fit_transform(dataset["month"])
dataset["day"] = label_encoder.fit_transform(dataset["day"])
dataset["size_category"] = label_encoder.fit_transform(dataset["size_category"])


# In[10]:


dataset.head(10)


# In[11]:


# split into input (X) and output (Y) variables
X = dataset.iloc[:,:11]


# In[12]:


Y = dataset.iloc[:,-1]


# In[13]:


X


# In[14]:


Y


# # Build ANN Model

# In[15]:


# create model
model = Sequential()
model.add(layers.Dense(50, input_dim=11,  activation='relu'))
model.add(layers.Dense(11,  activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


# In[16]:


# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])


# In[17]:


# Fit the model
history = model.fit(X, Y, validation_split=0.33, epochs=100, batch_size=10)


# In[18]:


# evaluate the model
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# # Visualization

# In[19]:


history.history.keys()


# In[20]:


# summarize history for accuracy
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[21]:


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:





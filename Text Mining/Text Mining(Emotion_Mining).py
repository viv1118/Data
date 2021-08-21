#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from scipy.stats import itemfreq
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer,HashingVectorizer
from sklearn.metrics import confusion_matrix


# # Import Dataset and Preprocessing

# In[6]:


data = pd.read_csv('text_emotion.csv',encoding = "ISO-8859-1")
data


# In[7]:


data.head(10)


# In[8]:


data.shape


# In[9]:


data.dtypes


# In[10]:


data.info()


# In[11]:


data.isnull().sum()


# In[14]:


new_data = data.rename(columns = {"sentiment": "Emotion"})
new_data


# # Visualization

# In[15]:


new_data['Emotion'].value_counts()


# In[17]:


import matplotlib.pyplot as plt
import seaborn as sns
new_data['Emotion'].value_counts().plot(kind='bar')


# In[18]:


sns.countplot(new_data['Emotion']) #old 
plt.figure(figsize=(20,10)) #new 
sns.countplot(x='Emotion', data=new_data)
plt.show()


# In[28]:


new_data1=new_data[['tweet_id','Emotion','content']].copy()


# In[29]:


new_data1.Emotion.value_counts()


# In[30]:


new_data1.Emotion = np.where((new_data1.Emotion == 'neutral') |(new_data1.Emotion == 'empty')|(new_data1.Emotion == 'boredom'),'neutral',new_data1.Emotion)


# In[31]:


new_data1.Emotion= np.where((new_data1.Emotion == 'fun') |(new_data1.Emotion == 'enthusiasm'),'fun',new_data1.Emotion)


# In[32]:


new_data1=new_data1[new_data1.Emotion !='neutral']


# In[33]:


new_data1.Emotion.value_counts()


# In[34]:


data2=pd.read_csv('tweets_clean.txt',sep='	',header=None)


# In[35]:


data2.head(10)


# In[36]:


data2.columns=['tweet_id','content','sentiment']


# In[38]:


data2.sentiment = data2.sentiment.str.replace(':: ','')


# In[41]:


new_data2 = data2.rename(columns = {"sentiment": "Emotion"})
new_data2   


# In[42]:


new_data2.Emotion.value_counts()


# In[44]:


data = new_data1.append(new_data2)
data.head(10)


# In[45]:


data.Emotion = np.where((data.Emotion == 'disgust') |(data.Emotion == 'hate'),'hate',data.Emotion)


# In[46]:


data.Emotion.value_counts()


# In[47]:


def Clean_text(data):
    tweets = []
    sentiments = []
    for index,row in data.iterrows():
        sentence = re.sub(pattern,'',row.text)
        words = [e.lower() for e in sentence.split()]
        words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
        words = ' '.join(words)
        tweets.append(words)
        sentiments.append(row.sentiment)
    return tweets,sentiments


# In[49]:


import matplotlib.pyplot as plt
from wordcloud import WordCloud
allwords = " ".join([twts for twts in data["Emotion"]])
wordCloud = WordCloud(width = 1000, height = 1000, random_state = 21, max_font_size = 119).generate(allwords)
plt.figure(figsize=(20, 20), dpi=80)
plt.imshow(wordCloud, interpolation = "bilinear")
plt.axis("off")
plt.show()


# # Clean Text

# In[50]:


data=data[data.Emotion.isin(['sadness','worry','joy'])]


# In[51]:


data.Emotion.value_counts()


# # Remove irrelevant characters other than alphanumeric and space

# In[52]:


data['content']=data['content'].str.replace('[^A-Za-z0-9\s]+', '')


# In[53]:


#Remove links from the text
data['content']=data['content'].str.replace('http\S+|www.\S+', '', case=False)


# In[54]:


#converting everything in lowercase
data['content']=data['content'].str.lower()


# In[55]:


data['content']


# # Assign Target Variable

# In[57]:


target=data.Emotion
data = data.drop(['Emotion'],axis=1)


# In[58]:


data


# In[59]:


le=LabelEncoder()


# In[60]:


target=le.fit_transform(target)


# In[61]:


target


# # Split Data into train & test

# In[62]:


X_train, X_test, y_train, y_test = train_test_split(data,target,stratify=target,test_size=0.4, random_state=42)


# In[66]:


np.unique(y_train,return_index=True)


# In[68]:


np.unique(y_test, return_index=True)


# # Tokenization
# Tokenization can be done in a variety of ways, namely Bag of words, tf-idf, Glove, word2vec ,fasttext etc. Lets see how they can be applied and how they affect the accuracy

# # Bag of Words

# In[69]:


count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train.content)
X_test_counts =count_vect.transform(X_test.content)
print('Shape of Term Frequency Matrix: ',X_train_counts.shape)


# # Naive Bayes Model

# In[70]:


clf = MultinomialNB().fit(X_train_counts,y_train)
predicted = clf.predict(X_test_counts)
nb_clf_accuracy = np.mean(predicted == y_test) * 100
print(nb_clf_accuracy)


# # Pipeline

# In[71]:


def print_acc(model):
    predicted = model.predict(X_test.content)
    accuracy = np.mean(predicted == y_test) * 100
    print(accuracy)


# In[72]:


nb_clf = Pipeline([('vect', CountVectorizer()), ('clf', MultinomialNB())])
nb_clf = nb_clf.fit(X_train.content,y_train)
print_acc(nb_clf)


# # TF IDF transformer

# In[73]:


nb_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
nb_clf = nb_clf.fit(X_train.content,y_train)
print_acc(nb_clf)


# In[74]:


confusion_matrix(y_test,predicted)


# # Remove Stop Words

# In[75]:


stop_words = set(stopwords.words('english'))
nb_clf = Pipeline([('vect', CountVectorizer(stop_words=stop_words)), ('clf', MultinomialNB())])
nb_clf = nb_clf.fit(X_train.content,y_train)
print_acc(nb_clf)


# In[76]:


nb_clf = Pipeline([('vect', CountVectorizer(stop_words=stop_words)), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
nb_clf = nb_clf.fit(X_train.content,y_train)
print_acc(nb_clf)


# # Lemmatization

# In[77]:


w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()


# In[84]:


def lemmatize_text(text):
    return ' '.join([lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)])


# In[85]:


X_train.loc[:,'content'] = X_train['content'].apply(lemmatize_text)


# In[86]:


X_test.loc[:,'content'] = X_test['content'].apply(lemmatize_text)


# In[87]:


nb_clf = Pipeline([('vect', CountVectorizer(stop_words=stop_words)), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
nb_clf = nb_clf.fit(X_train.content,y_train)
print_acc(nb_clf)


# In[ ]:





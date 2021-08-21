#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install spacy')


# In[2]:


get_ipython().system('pip install wordcloud')


# In[4]:


# execute below command through anaconda command prompt 
#python -m spacy download en_core_web_md


# # Import Libraries

# In[21]:


import pandas as pd
import numpy as np
import string
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from matplotlib.pyplot import imread
from wordcloud import WordCloud
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split


# # Load Dataset

# In[26]:


musk=pd.read_csv("Elon_musk.csv", sep=',', encoding='latin-1')
musk


# In[30]:


train,test = train_test_split(musk,test_size=0.1)


# In[31]:


train


# In[32]:


test


# In[54]:


import re

# Clean The Data
def cleantext(text):
    text = re.sub(r"@[A-Za-z0-9]+", "", text) # Remove Mentions
    text = re.sub(r"#", "", text) # Remove Hashtags Symbol
    text = re.sub(r"RT[\s]+", "", text) # Remove Retweets
    text = re.sub(r"https?:\/\/\S+", "", text) # Remove The Hyper Link
    
    return text

# Clean The Text
musk["Text"] = musk["Text"].apply(cleantext)

musk.head()


# In[59]:


from textblob import TextBlob

# Get The Subjectivity
def sentiment_analysis(ds):
    sentiment = TextBlob(ds["Text"]).sentiment
    return pd.Series([sentiment.subjectivity, sentiment.polarity])

# Adding Subjectivity & Polarity
musk[["subjectivity", "polarity"]] = musk.apply(sentiment_analysis, axis=1)
musk


# In[92]:


import matplotlib.pyplot as plt
from wordcloud import WordCloud
allwords = " ".join([twts for twts in musk["Text"]])
wordCloud = WordCloud(width = 1000, height = 1000, random_state = 21, max_font_size = 119).generate(allwords)
plt.figure(figsize=(20, 20), dpi=80)
plt.imshow(wordCloud, interpolation = "bilinear")
plt.axis("off")
plt.show()


# In[61]:


# Compute The Negative, Neutral, Positive Analysis
def analysis(score):
    if score < 0:
        return "Negative"
    elif score == 0:
        return "Neutral"
    else:
        return "Positive"


# In[62]:


# Create a New Analysis Column
musk["analysis"] = musk["polarity"].apply(analysis)


# In[63]:


# Print The Data
musk


# In[64]:


positive_tweets = musk[musk['analysis'] == 'Positive']
negative_tweets = musk[musk['analysis'] == 'Negative']

print('positive tweets')
for i, row in positive_tweets[:5].iterrows():
  print(' -' + row['Text'])

print('negative tweets')
for i, row in negative_tweets[:5].iterrows():
  print(' -' + row['Text'])


# In[66]:


musk = musk[['Text','analysis']]
musk


# In[68]:


musk = musk [musk ['analysis']!= 'Neutral']
musk


# In[69]:


musk.shape


# In[70]:


musk.dtypes


# In[71]:


musk.info()


# # Data Preprocessing

# In[72]:


train,test = train_test_split(musk,test_size=0.1)


# In[73]:


train


# In[74]:


test


# In[91]:


for val in train['Text']:
    print (val)


# In[114]:


ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()


# In[115]:


pattern = "(#\w+)|(RT\s@\w+:)|(http.*)|(@\w+)"


# In[119]:


train['Text'][1000]


# In[123]:


def Clean_Text(musk):
    Text = []
    analysis = []
    for index,row in musk.iterrows():
        sentence = re.sub(pattern,'',row.Text)
        words = [e.lower() for e in sentence.split()]
        words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
        words = ' '.join(words)
        Text.append(words)
        analysis.append(row.analysis)
    return Text,analysis


# In[124]:


train_Text,train_analysis = Clean_Text(train)


# In[126]:


final_data = {'tweets':train_Text,'sentiments':train_analysis}
final_data


# In[128]:


processed_data = pd.DataFrame(final_data)


# In[129]:


processed_data


# # Label Encoding

# In[130]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
processed_data['sentiments'] = labelencoder.fit_transform(processed_data['sentiments'])


# In[131]:


processed_data


# # Visualization

# In[132]:


import matplotlib.pyplot as plt
from wordcloud import WordCloud
allwords = " ".join([twts for twts in processed_data["tweets"]])
wordCloud = WordCloud(width = 1000, height = 1000, random_state = 21, max_font_size = 119).generate(allwords)
plt.figure(figsize=(20, 20), dpi=80)
plt.imshow(wordCloud, interpolation = "bilinear")
plt.axis("off")
plt.show()


# In[133]:


sns.distplot(processed_data['sentiments'])


# In[134]:


sns.boxplot(processed_data['sentiments'])


# # Converting Words into Vectors

# In[135]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(ngram_range=(1,3))
cv.fit(processed_data['tweets'])


# In[136]:


X_train = cv.transform(processed_data['tweets'])


# In[137]:


print(X_train.shape)


# In[138]:


X_train


# In[139]:


target = processed_data['sentiments'].values


# In[140]:


target


# # Sentiment Analysis (Model Building)

# In[141]:


from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()


# In[142]:


classifier.fit(X_train,target)


# In[144]:


test_Text,test_analysis = Clean_Text(test)


# In[145]:


data_test = {'tweets':test_Text,'sentiments':test_analysis}
final_test_data = pd.DataFrame(data_test)


# In[146]:


final_test_data


# In[147]:


X_test = cv.transform(final_test_data['tweets'])


# In[148]:


X_test


# In[149]:


X_test.shape


# In[151]:


y_pred = classifier.predict(X_test)
y_pred


# In[152]:


final_test_data['sentiments'] = labelencoder.fit_transform(final_test_data['sentiments'])


# In[153]:


final_test_data


# In[154]:


actual_values = final_test_data['sentiments'].values


# In[155]:


actual_values


# In[156]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred, actual_values))


# In[ ]:





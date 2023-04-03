#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import ast


# In[2]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[3]:


movies.head()


# In[4]:


credits.head(1)


# In[5]:


movies = movies.merge(credits,on='title')


# In[6]:


movies = movies[['id','title','overview','genres','keywords','cast','crew']]


# In[7]:


movies.head()


# In[8]:


movies.isnull().sum()


# In[9]:


movies.dropna(inplace=True)


# In[10]:


movies.isnull().sum()


# In[11]:


movies.duplicated().sum()


# In[12]:


movies.iloc[0].genres


# In[13]:


def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[14]:


movies.genres = movies.genres.apply(convert)


# In[15]:


movies.keywords = movies.keywords.apply(convert)


# In[16]:


movies.head()


# In[17]:


def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else: break
    return L


# In[18]:


movies.cast = movies.cast.apply(convert3)


# In[19]:


movies.head()


# In[20]:


def fetch_dir(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L


# In[21]:


movies.crew = movies.crew.apply(fetch_dir)


# In[22]:


movies.overview = movies.overview.apply(lambda x : x.split())


# In[23]:


movies.head()


# In[24]:


movies['genres'] = movies['genres'].apply(lambda x : [i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x : [i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x : [i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x : [i.replace(" ","") for i in x])


# In[25]:


movies.head()


# In[26]:


movies['tags'] = movies.overview + movies.genres + movies.keywords + movies.cast + movies.crew


# In[27]:


new_df = movies[['id','title','tags']]


# In[28]:


new_df.tags=new_df.tags.apply(lambda x:" ".join(x))


# In[29]:


new_df.tags=new_df.tags.apply(lambda x:x.lower())


# In[30]:


new_df.head()


# In[31]:


pip install nltk
import nltk


# In[33]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[34]:


def stem(text):
    L = []
    for i in text.split():
        L.append(ps.stem(i))
    return " ".join(L)


# In[35]:


new_df.tags = new_df.tags.apply(stem)


# In[36]:


pip install scikit-learn


# In[37]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[38]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[39]:


from sklearn.metrics.pairwise import cosine_similarity


# In[40]:


similarity = cosine_similarity(vectors)


# In[41]:


def recommend(movie):
    movie_index = new_df[new_df.title == movie.title()].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key = lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# In[42]:


recommend('batman')


# In[43]:


new_df.head()


# In[44]:


import pickle


# In[45]:


pickle.dump(new_df,open('movies.pkl','wb'))


# In[46]:


pickle.dump(similarity,open('similarity.pkl','wb'))


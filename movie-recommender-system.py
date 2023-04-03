#!/usr/bin/env python
# coding: utf-8

# In[32]:


import numpy as np
import pandas as pd
import ast
import gzip
import nltk


# In[33]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[34]:


movies.head()


# In[35]:


credits.head(1)


# In[36]:


movies = movies.merge(credits,on='title')


# In[37]:


movies = movies[['id','title','overview','genres','keywords','cast','crew']]


# In[38]:


movies.head()


# In[39]:


movies.isnull().sum()


# In[40]:


movies.dropna(inplace=True)


# In[41]:


movies.isnull().sum()


# In[42]:


movies.duplicated().sum()


# In[43]:


movies.iloc[0].genres


# In[44]:


def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[45]:


movies.genres = movies.genres.apply(convert)


# In[46]:


movies.keywords = movies.keywords.apply(convert)


# In[47]:


movies.head()


# In[48]:


def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else: break
    return L


# In[49]:


movies.cast = movies.cast.apply(convert3)


# In[50]:


movies.head()


# In[51]:


def fetch_dir(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L


# In[52]:


movies.crew = movies.crew.apply(fetch_dir)


# In[53]:


movies.overview = movies.overview.apply(lambda x : x.split())


# In[54]:


movies.head()


# In[55]:


movies['genres'] = movies['genres'].apply(lambda x : [i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x : [i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x : [i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x : [i.replace(" ","") for i in x])


# In[56]:


movies.head()


# In[57]:


movies['tags'] = movies.overview + movies.genres + movies.keywords + movies.cast + movies.crew


# In[58]:


new_df = movies[['id','title','tags']]


# In[59]:


new_df.tags=new_df.tags.apply(lambda x:" ".join(x))


# In[60]:


new_df.tags=new_df.tags.apply(lambda x:x.lower())


# In[61]:


new_df.head()


# In[62]:


pip install nltk


# In[63]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[64]:


def stem(text):
    L = []
    for i in text.split():
        L.append(ps.stem(i))
    return " ".join(L)


# In[65]:


new_df.tags = new_df.tags.apply(stem)


# In[66]:


pip install scikit-learn


# In[67]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[68]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[69]:


from sklearn.metrics.pairwise import cosine_similarity


# In[70]:


similarity = cosine_similarity(vectors)


# In[71]:


def recommend(movie):
    movie_index = new_df[new_df.title == movie.title()].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key = lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# In[72]:


recommend('batman')


# In[73]:


new_df.head()


# In[74]:


import pickle


# In[75]:


pickle.dump(new_df,open('movies.pkl','wb'))


# In[76]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[77]:


output_file_path = "similarity.pkl.gz"

# Open the output gzip file in write-binary mode
with gzip.open(output_file_path, "wb") as output_file:
    # Dump the data to the output gzip file in pickle format
    pickle.dump(similarity, output_file, protocol=pickle.HIGHEST_PROTOCOL)


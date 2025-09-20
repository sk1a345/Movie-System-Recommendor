import ast

import pandas as pd
import numpy as np

credits = pd.read_csv(r"C:\Users\HP\OneDrive\Machine_Learning_projects\Movie-System-Recommender\data\tmdb_5000_credits.csv")
movies =  pd.read_csv(r"C:\Users\HP\OneDrive\Machine_Learning_projects\Movie-System-Recommender\data\tmdb_5000_movies.csv")

# print(credits.head())
# print(movies.head())

movies = movies.merge(credits, on='title')
# print(movies.head())
# print(movies.info())
# print(movies.shape)

# what columns to keep: genres, id, keywords, overview, title,cast,crew

movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
# print(movies.info())

# handling the missing values:
# print(movies.isnull().sum())

# Dropping the rows with the missing values:
# print(movies.shape)
movies.dropna(inplace=True)
# print(movies.shape)

# Searching for the duplicate data:
# print(movies.duplicated().sum())

# print(movies.iloc[0].genres)

# helper function:
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
# print(movies['genres'])

# print(movies.iloc[0].keywords)
# print(movies.iloc[0].title)

movies['keywords'] = movies['keywords'].apply(convert)

# print(movies['keywords'])

# print(movies['cast'][0])

def convert3(obj):
    L = []
    counter =0
    for i in ast.literal_eval(obj):
        if counter !=3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L

movies['cast'] = movies['cast'].apply(convert3)
# print(movies['cast'])

# for crew:
def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L
movies['crew'] = movies['crew'].apply(fetch_director)
# print(movies['crew'])

# print(movies.iloc[0].overview)


movies['overview'] = movies['overview'].apply(lambda x: x.split())

# print(movies['overview'])

movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ","") for i in x])
movies['keywords'] =movies['keywords'].apply(lambda x: [i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","")for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","")for i in x])

# print(movies.iloc[5])
# print(movies['genres'])

movies['tags'] = movies['overview'] +movies['genres']+movies['keywords']+movies['cast']+movies['crew']
# print(movies['tags'])

new_df = movies[['movie_id','title','tags']]
# print(new_df)

new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

# print(new_df['tags'][0])

new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
# print(new_df['tags'][0])

import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

# print(ps.stem("loving"))
# print(ps.stem("love"))
# print(ps.stem("loved"))
# print(ps.stem("Dancing"))

# print(new_df['tags'].apply(stem))

new_df['tags'] = new_df['tags'].apply(stem)

# print(new_df['tags'][0])



# Vecotrization :
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words = 'english')
vectors = cv.fit_transform(new_df['tags']).toarray()

# print(vectors)


from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)
# print(similarity.shape)

def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    for i in movie_list:
        print(new_df.iloc[i[0]].title)

# recommend('Spectre')

import pickle

# pickle.dump(new_df,open('movies.pkl','wb'))

pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))

pickle.dump(similarity,open('similarity.pkl','wb'))

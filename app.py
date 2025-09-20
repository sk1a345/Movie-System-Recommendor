import streamlit as st
import pickle
import pandas as pd


def recommend(movie):
    movie_ind = movies[movies['title']==movie].index[0]
    distances = similarity[movie_ind]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x: x[1])[1:6]

    recommended_movies = []
    for i in movies_list:
        recommended_movies.append(movies.iloc[i[0]].title)
    return recommended_movies

movies_dict = pickle.load(open("movie_dict.pkl",'rb'))
similarity = pickle.load(open("similarity.pkl",'rb'))

movies = pd.DataFrame(movies_dict)

st.title("Movie Recommender System")

selected_movie_name = st.selectbox(
"Hello please select one movie",
movies['title'].values
)

# Creating button:
if st.button("Recommed"):
    recommendations = recommend(selected_movie_name)
    for i in recommendations:
        st.write(i)

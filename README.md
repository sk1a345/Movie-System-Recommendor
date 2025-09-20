🎬 Movie Recommender System

A content-based movie recommendation system built with Python and Streamlit. This app recommends movies similar to the one you select using genres, keywords, cast, crew, and overview.

It uses NLP techniques and cosine similarity to find movies with similar tags and displays them interactively using Streamlit.

📝 Features

Recommend top 5 movies similar to your selection

Preprocessed movie data with genres, cast, crew, keywords, and overview

Clean and simple Streamlit web interface

Data and similarity matrices stored using pickle for fast performance

💻 Technologies Used

Python 3.x

Pandas, NumPy

NLTK (Natural Language Toolkit) for text preprocessing

Scikit-learn (CountVectorizer & Cosine Similarity)

Streamlit for interactive web app

📁 Dataset

The app uses the TMDB 5000 Movies Dataset:

tmdb_5000_movies.csv.zip

tmdb_5000_credits.csv.zip

These CSV files are extracted, preprocessed, and combined for recommendation purposes.

⚡ How It Works

Data Preprocessing:

Merge movies and credits data

Keep only relevant columns (title, genres, keywords, cast, crew, overview)

Remove missing values and duplicates

Convert JSON-like strings to Python lists

Clean text, remove spaces in multi-word names, and lowercase

Combine features into a single tags column

Apply stemming using NLTK

Feature Extraction:

Convert tags into vectors using CountVectorizer

Compute similarity using cosine similarity

Recommendation:

Select a movie

Find the top 5 movies with the highest similarity score

Display recommendations in Streamlit

🚀 Setup & Installation

Clone the repository

git clone https://github.com/yourusername/movie-recommender.git
cd movie-recommender


Create and activate a virtual environment (optional but recommended)

python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate


Install dependencies

pip install -r requirements.txt


(Dependencies include: pandas, numpy, nltk, scikit-learn, streamlit)

Download the dataset

Place the following files in the data folder:

tmdb_5000_movies.csv.zip

tmdb_5000_credits.csv.zip

Run the preprocessing script

python movie_system_recommendation.py


This will generate:

movie_dict.pkl

similarity.pkl

Run the Streamlit app

streamlit run app.py


Open the link provided by Streamlit in your browser to start using the app.

📌 Usage

Select a movie from the dropdown menu

Click the Recommend button

The app displays the top 5 recommended movies

🛠 Folder Structure
Movie-Recommender/
│
├── app.py                     # Streamlit application
├── movie_system_recommendation.py  # Preprocessing & similarity calculation
├── extract_and_load.py        # Optional CSV extraction script
├── movie_dict.pkl             # Pickle file of processed movie data
├── similarity.pkl             # Pickle file of similarity matrix
├── data/                      # Folder containing datasets
│   ├── tmdb_5000_movies.csv.zip
│   └── tmdb_5000_credits.csv.zip
├── README.md
└── requirements.txt

📌 Dependencies (requirements.txt)
pandas
numpy
nltk
scikit-learn
streamlit

📚 References

TMDB 5000 Movies Dataset

NLTK Documentation

Scikit-learn Documentation

Streamlit Documentation

import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# Load datasets
# -------------------------
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

# Merge datasets
movies = movies.merge(credits, on='title')

# -------------------------
# Select important columns
# -------------------------
movies = movies[['movie_id', 'title', 'genres', 'keywords', 'cast', 'crew']]

# -------------------------
# Convert string JSON to list
# -------------------------
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

# Top 3 cast only
def convert_cast(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

movies['cast'] = movies['cast'].apply(convert_cast)

# Director from crew
def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L

movies['crew'] = movies['crew'].apply(fetch_director)

# -------------------------
# Clean data (remove spaces)
# -------------------------
def collapse(L):
    return [i.replace(" ", "") for i in L]

movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)
movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)

# -------------------------
# Create tags column
# -------------------------
movies['tags'] = movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# Convert list to string
movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))

# Final dataset
new_df = movies[['movie_id', 'title', 'tags']]

# Lowercase
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

# -------------------------
# Vectorization
# -------------------------
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

# -------------------------
# Similarity
# -------------------------
similarity = cosine_similarity(vectors)

# -------------------------
# Recommendation function
# -------------------------
def recommend(movie):
    movie = movie.lower()
    new_df['title'] = new_df['title'].str.lower()

    if movie not in new_df['title'].values:
        print("Movie not found!")
        return

    index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[index]

    movie_list = sorted(list(enumerate(distances)),
                        reverse=True,
                        key=lambda x: x[1])[1:6]

    print("\nRecommended Movies:")
    for i in movie_list:
        print(new_df.iloc[i[0]].title)

# -------------------------
# User input
# -------------------------
movie_name = input("Enter movie name: ")
recommend(movie_name)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import pickle
from tensorflow.keras.models import load_model

from NCF import NeuralCollaborativeFiltering

with open('M:/Movie-Recommendation-Engine/notebooks/user_mapping.pkl', 'rb') as f:
    user_mapping = pickle.load(f)

with open('M:/Movie-Recommendation-Engine/notebooks/item_mapping.pkl', 'rb') as f:
    item_mapping = pickle.load(f)

n_users = len(user_mapping)
n_items = len(item_mapping)

model = NeuralCollaborativeFiltering(n_users=n_users, n_items=n_items)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.load_weights('./model.weights.h5')

def get_user_input():
    print("\nEnter the IDs of three movies you have watched and their ratings (1-5):")
    user_ratings = []
    for i in range(3):
        try:
            movie_id = int(input(f"Movie {i + 1} ID: "))
            rating = float(input(f"Movie {i + 1} Rating (0.0 - 5.0): "))
            user_ratings.append((movie_id, rating))
        except ValueError:
            print("Invalid input. Please try again.")
            return get_user_input()
    return user_ratings

def recommend_movies(user_ratings, model, movies_df, top_n=10):
    placeholder_user_id = 0

    seen_movie_ids = [movie_id for movie_id, _ in user_ratings]
    unseen_movies = movies_df.loc[~movies_df['movieId'].isin(seen_movie_ids), ['movieId', 'title']]
    unseen_movies['mappedId'] = unseen_movies['movieId'].map(item_mapping)
    unseen_movies = unseen_movies.dropna(subset=['mappedId']).astype({'mappedId': int})

    user_input = np.full(len(unseen_movies), placeholder_user_id, dtype=int)
    item_input = unseen_movies['mappedId'].to_numpy()

    predictions = model.predict([user_input, item_input], batch_size=512)
    unseen_movies['prediction'] = predictions.flatten()
    recommendations = unseen_movies.sort_values(by='prediction', ascending=False).head(top_n)

    return recommendations

def main():
    while True:
        print("\nWelcome to the Movie Recommendation System!")
        user_ratings = get_user_input()
        recommendations = recommend_movies(user_ratings, model, movies_df)
        print("\nTop Recommendations:")
        print(recommendations[['title', 'prediction']])
        another = input("\nGet another recommendation? (y/n): ").strip().lower()
        if another != 'y':
            break

if __name__ == "__main__":
    movies_df = pd.read_csv(
        'M:/Movie-Recommendation-Engine/data/raw/ml-1m/movies.dat', 
        sep='::', 
        engine='python', 
        names=['movieId', 'title', 'genres'], 
        encoding='latin-1'
    )
    main()
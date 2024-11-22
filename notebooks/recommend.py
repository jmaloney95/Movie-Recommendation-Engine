import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import pickle
from tensorflow.keras.models import load_model
import import_ipynb

from NCF.ipynb import NeuralCollaborativeFiltering

with open('notebooks/NCF Local/user_mapping.pkl', 'rb') as f:
    user_mapping = pickle.load(f)

with open('notebooks/NCF Local/item_mapping.pkl', 'rb') as f:
    item_mapping = pickle.load(f)

model = NeuralCollaborativeFiltering(n_users=n_users, n_items=n_items)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.load_weights('notebooks/model.weights.h5')

def get_user_input():
    print("Enter the IDs of three movies you have watched and their ratings:")
    user_ratings = []
    for i in range(3):
        movie_id = int(input(f"Movie {i+1} ID: "))
        rating = float(input(f"Movie {i+1} Rating (0.0 - 5.0): "))
        user_ratings.append((movie_id, rating))
    return user_ratings

def recommend_movies(user_ratings, model, movies_df, top_n=10):
    new_user_id = len(user_mapping)
    seen_movie_ids = [movie_id for movie_id, _ in user_ratings]
    unseen_movie_ids = movies_df.loc[~movies_df['movieId'].isin(seen_movie_ids), 'movieId']
    
    user_input = [new_user_id] * len(unseen_movie_ids)
    item_input = unseen_movie_ids.map(item_mapping).values

    predictions = model.predict([user_input, item_input])
    recommendations = pd.DataFrame({
        'movieId': unseen_movie_ids.values,
        'prediction': predictions.flatten()
    }).sort_values(by='prediction', ascending=False).head(top_n)

    if 'title' in movies_df.columns:
        recommendations = recommendations.merge(movies_df[['movieId', 'title']], on='movieId', how='left')

    return recommendations

if __name__ == "__main__":
    movies_df = pd.read_csv('M:/Movie-Recommendation-Engine/data/raw/ml-1m/movies.dat')
    user_ratings = get_user_input()
    recommendations = recommend_movies(user_ratings, model, movies_df)
    print("Top Recommendations:")
    print(recommendations)
    recommendations.plot(kind='bar', x='title', y='prediction', legend=False)
    plt.title("Top Movie Recommendations")
    plt.ylabel("Predicted Rating")
    plt.show()
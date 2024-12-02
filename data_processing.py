import pandas as pd
import pickle

def load_mappings(user_mapping_path, item_mapping_path):
    with open(user_mapping_path, 'rb') as f:
        user_mapping = pickle.load(f)
    with open(item_mapping_path, 'rb') as f:
        item_mapping = pickle.load(f)
    return user_mapping, item_mapping

def load_movies_data(filepath):
    movies_df = pd.read_csv(
        filepath, 
        sep='::', 
        engine='python', 
        names=['movieId', 'title', 'genres'], 
        encoding='latin-1'
    )
    return movies_df
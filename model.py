import tensorflow as tf
from tensorflow.keras.models import load_model
from NCF import NeuralCollaborativeFiltering

def load_trained_model(n_users, n_items, weights_filepath):
    model = NeuralCollaborativeFiltering(n_users=n_users, n_items=n_items)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.load_weights(weights_filepath)
    return model

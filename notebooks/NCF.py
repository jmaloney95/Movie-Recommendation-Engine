#!/usr/bin/env python
# coding: utf-8

# Import required libraries
import tensorflow as tf
from tensorflow.keras import layers, Model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pickle
from sklearn.model_selection import train_test_split
from recommenders.evaluation.python_evaluation import (
    map, ndcg_at_k, precision_at_k, recall_at_k
)
from keras.saving import register_keras_serializable
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam, SGD

print("TensorFlow version:", tf.__version__)

# Constants
TOP_K = 10  # Top-K items for evaluation
EPOCHS = 100  # Number of training epochs
BATCH_SIZE = 256  # Batch size for training
CHECKPOINT_FILEPATH = './model.weights.h5'  # Filepath to save model weights

# Load the dataset
ratings = pd.read_csv('M:/Movie-Recommendation-Engine/notebooks/NCF Local/NCF/ml-1m_dataset.csv')

# Split data into training and testing sets
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

# Map user and item IDs to continuous indices
user_mapping = {user_id: idx for idx, user_id in enumerate(train_data['userId'].unique())}
item_mapping = {item_id: idx for idx, item_id in enumerate(train_data['movieId'].unique())}

# Apply the mappings to training and testing data
train_data['userId'] = train_data['userId'].map(user_mapping)
train_data['movieId'] = train_data['movieId'].map(item_mapping)

test_data['userId'] = test_data['userId'].map(user_mapping)
test_data['movieId'] = test_data['movieId'].map(item_mapping)

# Handle missing values in the test set
test_data = test_data.copy()
test_data['userId'] = test_data['userId'].fillna(0).astype(int)
test_data['movieId'] = test_data['movieId'].fillna(0).astype(int)

# Count the number of users and items
n_users = len(user_mapping)
n_items = len(item_mapping)
print(f"Users: {n_users}, Items: {n_items}")

# Define the Neural Collaborative Filtering model
@register_keras_serializable(package="Custom", name="NeuralCollaborativeFiltering")
class NeuralCollaborativeFiltering(Model):
    def __init__(self, n_users, n_items, embedding_dim=8):
        super(NeuralCollaborativeFiltering, self).__init__()
        # Embedding layers for users and items
        self.user_embedding = layers.Embedding(n_users, embedding_dim)
        self.item_embedding = layers.Embedding(n_items, embedding_dim)
        
        # Fully connected layers
        self.dense_layers = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
        ])

    def call(self, inputs):
        # Forward pass
        user_input, item_input = inputs
        user_emb = self.user_embedding(user_input)
        item_emb = self.item_embedding(item_input)
        concatenated = tf.concat([user_emb, item_emb], axis=-1)
        return self.dense_layers(concatenated)

print("Model defined successfully.")

# Training script
if __name__ == "__main__":
    
    def build_model(n_users, n_items):
        # Build and compile the NCF model
        model = NeuralCollaborativeFiltering(n_users=n_users, n_items=n_items)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    RESUME_TRAINING = False  # Whether to resume training from checkpoint

    # Optimizers to test
    optimizers = {
        'Adam': Adam(learning_rate=0.001),
        'SGD with Momentum': SGD(learning_rate=0.01, momentum=0.9)
    }
    results = {}  # Store results for each optimizer
    
    for opt_name, optimizer in optimizers.items():
        print(f"\nTraining with {opt_name} optimizer...\n")

        model = build_model(n_users, n_items)

        # Load weights if resuming training
        if RESUME_TRAINING and os.path.exists(CHECKPOINT_FILEPATH):
            print("Checkpoint found. Resuming training...")
            model.load_weights(CHECKPOINT_FILEPATH)
        else:
            print("No checkpoint found. Training from scratch.")

        # Define a callback to save model weights during training
        checkpoint_callback = ModelCheckpoint(
            filepath=CHECKPOINT_FILEPATH,
            save_weights_only=True,
            save_best_only=False,
            verbose=1
        )

        # Prepare training data
        user_input = train_data['userId'].values
        item_input = train_data['movieId'].values
        labels = train_data['rating'].values > 3.5  # Binary labels: 1 if rating > 3.5, else 0

        # Start training
        start_time = time.time()
        history = model.fit(
            [user_input, item_input],
            labels,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.2,
            callbacks=[checkpoint_callback]
        )
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Total training time: {elapsed_time:.2f} seconds")

        # Plot training and validation accuracy
        train_acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        epochs = range(1, len(train_acc) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_acc, label='Training Accuracy')
        plt.plot(epochs, val_acc, label='Validation Accuracy', linestyle='--')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title(f'Training and Validation Accuracy with {opt_name}')
        plt.legend()
        plt.grid(True)

        # Save the training graph
        plot_path = f'training_validation_accuracy_{opt_name}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Training graph saved: {plot_path}")

        plt.show()

        # Save mappings
        with open('user_mapping.pkl', 'wb') as f:
            pickle.dump(user_mapping, f)
        with open('item_mapping.pkl', 'wb') as f:
            pickle.dump(item_mapping, f)

        # Evaluate the model
        user_input_test = test_data['userId'].values
        item_input_test = test_data['movieId'].values
        predictions = model.predict([user_input_test, item_input_test])
        test_data['prediction'] = predictions

        eval_map = map(test_data, test_data, col_prediction='prediction', k=TOP_K)
        eval_ndcg = ndcg_at_k(test_data, test_data, col_prediction='prediction', k=TOP_K)
        eval_precision = precision_at_k(test_data, test_data, col_prediction='prediction', k=TOP_K)
        eval_recall = recall_at_k(test_data, test_data, col_prediction='prediction', k=TOP_K)

        print(
            f"MAP: {eval_map:.6f}\n"
            f"NDCG: {eval_ndcg:.6f}\n"
            f"Precision@K: {eval_precision:.6f}\n"
            f"Recall@K: {eval_recall:.6f}"
        )

        # Save the trained model
        model.save('M:/Movie-Recommendation-Engine/models/ncf_model.keras')
        print("Model saved as 'ncf_model.keras'")

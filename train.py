import os
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
from model import load_trained_model
from data_processing import load_mappings

# Constants
EPOCHS = 100
BATCH_SIZE = 256
CHECKPOINT_FILEPATH = './model.weights.h5'

def load_training_data():
    # Load dataset
    ratings = pd.read_csv('data/raw/ml-1m/ml-1m_dataset.csv')
    train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

    # Create mappings
    user_mapping = {user_id: idx for idx, user_id in enumerate(train_data['userID'].unique())}
    item_mapping = {item_id: idx for idx, item_id in enumerate(train_data['itemID'].unique())}

    # Map user and item IDs
    train_data['userID'] = train_data['userID'].map(user_mapping)
    train_data['itemID'] = train_data['itemID'].map(item_mapping)

    test_data['userID'] = test_data['userID'].map(user_mapping)
    test_data['itemID'] = test_data['itemID'].map(item_mapping)

    n_users = len(user_mapping)
    n_items = len(item_mapping)

    # Save mappings
    with open('notebooks/user_mapping.pkl', 'wb') as f:
        pickle.dump(user_mapping, f)
    with open('notebooks/item_mapping.pkl', 'wb') as f:
        pickle.dump(item_mapping, f)

    return train_data, test_data, n_users, n_items

def train_model():
    train_data, test_data, n_users, n_items = load_training_data()

    # Build model
    model = load_trained_model(n_users, n_items, CHECKPOINT_FILEPATH)

    # Prepare data for training
    user_input = train_data['userID'].values
    item_input = train_data['itemID'].values
    labels = train_data['rating'].values > 3.5

    # Define checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        filepath=CHECKPOINT_FILEPATH,
        save_weights_only=True,
        save_best_only=False,
        verbose=1
    )

    # Train the model
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

    # Save training graph
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(1, len(train_acc) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_acc, label='Training Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_validation_accuracy.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    train_model()

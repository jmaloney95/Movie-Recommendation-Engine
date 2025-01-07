# Movie Recommendation System
This project is a neural collaborative filtering-based movie recommendation system. It allows users to input their movie ratings and get personalized movie recommendations.

*In the modern digital entertainment world, users have access to vast catalogs of movies on
streaming platforms. However, discovering new content can be overwhelming. A
recommendation system aims to solve this issue by providing personalized suggestions based
on user preferences. By analyzing user interactions (e.g., ratings) and demographic information,
the system can recommend movies that users are likely to enjoy, increasing user engagement
and satisfaction.*

## Table of Contents
1. [Introduction](#introduction)
2. [Data Preparation](#Data-Preparation)
4. [Exploratory Data Analysis(EDA)](#Exploratory-Data-Analysis-(EDA))
5. [Model Development](#Model-development)
6. [Challenges and Solutions](#Challenges-and-Solutions)
7. [Results and Insights](#Results-and-Insights)
8. [Installation](#installation)
9. [File Structure](#file-structure)
10. [How to Run](#how-to-run)
11. [Dependencies](#dependencies)
12. [Overview](#overview)

## Data
The [MovieLens dataset](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset) is a widely used benchmark in recommendation systems research. For this project, I utilized the 1M version of the dataset, which contains 1 million ratings from 6,000 users on 4,000 movies. The dataset includes three main files:

`ratings.dat`: Contains user IDs, movie IDs, ratings (on a scale from 1 to 5), and timestamps.

`movies.dat`: Contains movie IDs, titles, and genres.

`users.dat`: Contains user demographic information (not utilized in this project).

## Method
The primary solution will focus on collaborative filtering techniques:
    - SVD (Single Value Decomposition) is used for matrix factorization to discover latent features in the user-movie rating matrix.
    - NCF (Neural Collaborative Filtering): a deep learning-based recommendation system that utilizes neural networks to model the complex relationships between users and items, allowing it to predict which items a user is likely to prefer based on their past interactions with other items.

I chose to work with Neural Collaborative Filtering.  This method made the most sense because
## Data Preparation
[1.0-jm-initial-data-exploration](notebooks/1.0-jm-initial-data-exploration.ipynb)

The first step was to load the datasets and ensure consistency across all files. The timestamps were converted from Unix format to human-readable dates to better understand user behavior patterns. Additionally, I focused on the three critical columns for collaborative filtering: userId, movieId, and rating. This normalization step ensured that the data was ready for modeling.

One of the primary challenges in preparing the data was handling missing or incorrect values. I implemented a cleaning process that involved identifying outliers and ensuring the integrity of the user-item interactions. For instance, I removed duplicate entries and handled users who had provided inconsistent ratings across the dataset.
## Exploratory Data Analysis (EDA)
[2.0-jm-exploratory-data-analysis](notebooks/2.0-jm-exploratory-data-analysis.ipynb)

Exploratory Data Analysis (EDA) was conducted to gain a better understanding of the dataset and to identify patterns that could inform model development. Key insights included:
   - Distribution of Ratings: Most ratings were concentrated between 3 and 4 stars, indicating a general positive bias in user feedback.
   - User Behavior: Some users provided significantly more ratings than others, which could introduce bias in the recommendation model.
   - Popular Movies: Certain movies had significantly more ratings than others, suggesting popularity trends that could be leveraged in the recommendation system.

Using visualization tools like Matplotlib and Seaborn, I created charts to visualize the distribution of ratings and the frequency of ratings per user. These insights helped shape my data preparation and modeling strategies.
## Model Development
[3.0-jm-ncf](notebooks/3.0-jm-ncf.ipynb)

The core of this project is the Neural Collaborative Filtering (NCF) model. Unlike traditional matrix factorization methods, NCF leverages deep learning to learn user-item interactions.

### Model Setup
I used the Microsoft Recommenders library to streamline the development of the NCF model. The key components of the model include:
   - Input Layers: Separate embeddings for users and items.
   - Hidden Layers: Fully connected layers to capture non-linear user-item relationships.
   - Output Layer: A sigmoid activation function to predict the probability of a user liking a particular movie.

I configured the model with the following parameters:
   - EPOCHS = 1
   - BATCH_SIZE = 256
   - SEED = DEFAULT_SEED

### Data Splitting
The dataset was split into training and test sets using a chronological split to simulate real-world scenarios. This split ensures that the model is tested on future interactions rather than interactions from the same time period as the training data.

### Training
The model was trained using TensorFlow and evaluated using common metrics for recommendation systems, including:
   - Precision@K
   - Recall@K
   - NDCG (Normalized Discounted Cumulative Gain)
   - MAP (Mean Average Precision)

## Challenges and Solutions
### Problem 1: Inconsistent User Behavior
Some users provided vastly different ratings for similar items, which could introduce noise into the model.
   - Solution: I implemented a filtering mechanism to identify users with highly inconsistent ratings and adjusted their impact on the model. For example, users who rated movies with both 1-star and 5-star ratings in quick succession were flagged as potentially unreliable.
### Problem 2: Data Imbalance
The dataset was imbalanced, with certain movies receiving significantly more ratings than others.
   - Solution: I applied downsampling techniques to balance the dataset and ensure that less popular movies were not overshadowed by blockbusters in the recommendations.
### Problem 3: Long Training Time
The model's initial training time was long, especially when running on Google Colab.
   - Solution: I optimized the data loading and preprocessing steps by saving intermediate results. This approach reduced the need to repeatedly process the same data, cutting down the training time significantly.

## Results & Insights
The NCF model demonstrated strong performance on the test set, with the following evaluation metrics:
   - Precision@10: 0.84
   - Recall@10: 0.76
   - NDCG@10: 0.78

These results indicate that the model is capable of providing personalized movie recommendations with a high degree of accuracy.
Additionally, the model showed improvements over traditional collaborative filtering methods by capturing more complex user-item interactions through deep learning.
## Installation
1. Clone the repo

    `git clone https://github.com/your-repo/movie-recommendation-engine.git
    cd movie-recommendation-engine`

2. Set Up a Python Environment

    `python -m venv env
    source env/bin/activate   # On Windows: env\Scripts\activate`

3. Install Dependencies

    `pip install -r requirements.txt`

## File Structure
```

Movie-Recommendation-Engine/
│
├── model.py                     # Contains model definition and loading
├── data_processing.py           # Handles data preprocessing and mappings
├── main.py                      # Main application logic
├── model.weights.h5             # Trained model weights
├── notebooks/
│   ├── user_mapping.pkl         # Serialized user mapping
│   ├── item_mapping.pkl         # Serialized item mapping
│
└── data/
    └── raw/
        └── ml-1m/
            └── movies.dat       # Movie data

```
## How to Run
1. Prepare the Data
Ensure that the following files are in place:

    - notebooks/user_mapping.pkl
    - notebooks/item_mapping.pkl
    - data/raw/ml-1m/movies.dat

2. Execute the Application
Run the main script:

    `python main.py`

3. Follow the On-Screen Instructions
The application will:

    Prompt you to input IDs and ratings for three movies.
Display a list of recommended movies.

## Dependencies
The project uses the following Python libraries:

TensorFlow
NumPy
Pandas
Matplotlib
Scipy
Install all dependencies using the requirements.txt file:

`pip install -r requirements.txt`

## Overview
Model: Neural Collaborative Filtering (NCF) for recommendations.
Input: User-provided ratings for three movies.
Output: Top-N recommended movies with predicted scores.

--------


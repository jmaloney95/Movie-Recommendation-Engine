# Movie Recommendation System
This project is a neural collaborative filtering-based movie recommendation system. It allows users to input their movie ratings and get personalized movie recommendations.
## Table of Contents
1. Installation
2. File Structure
3. How to Run
4. Dependencies
5. Overview

## Installation
1. Clone the repo

git clone https://github.com/your-repo/movie-recommendation-engine.git
cd movie-recommendation-engine

2. Set Up a Python Environment

python -m venv env
source env/bin/activate   # On Windows: env\Scripts\activate

3. Install Dependencies

pip install -r requirements.txt

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

notebooks/user_mapping.pkl
notebooks/item_mapping.pkl
data/raw/ml-1m/movies.dat

2. Execute the Application
Run the main script:
python main.py

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

pip install -r requirements.txt

## Overview
Model: Neural Collaborative Filtering (NCF) for recommendations.
Input: User-provided ratings for three movies.
Output: Top-N recommended movies with predicted scores.

--------


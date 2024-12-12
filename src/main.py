import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def limit_dataset_size(movie_data, sample_size=5000):
    """
    Randomly sample a smaller subset of the dataset to limit memory usage.

    Args:
        movie_data (pd.DataFrame): The full movie dataset.
        sample_size (int): The number of rows to sample.

    Returns:
        pd.DataFrame: A subset of the movie dataset.
    """
    if len(movie_data) > sample_size:
        # Randomly sample the dataset
        movie_data = movie_data.sample(n=sample_size, random_state=42).reset_index(drop=True)
    
    return movie_data

def clean_data(movie_data):
    """
    Clean the dataset by handling missing descriptions.

    Args:
        movie_data (pd.DataFrame): The movie dataset.

    Returns:
        pd.DataFrame: Cleaned dataset.
    """
    # Drop rows where 'description' column is NaN
    movie_data = movie_data.dropna(subset=['description'])
    # Alternatively, you can fill NaN values with a placeholder text:
    # movie_data['description'].fillna('No description available', inplace=True)
    
    return movie_data

def calculate_similarity(corpus):
    """
    Calculate the cosine similarity between movie descriptions in the corpus.

    Args:
        corpus (list of str): A list of movie descriptions.

    Returns:
        scipy.sparse.matrix: Matrix of pairwise cosine similarity scores.
    """
    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=2000)  # Limit the number of features

    # Convert descriptions to TF-IDF vectors
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Calculate cosine similarity between each pair of movie descriptions
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return similarity_matrix

# Usage example:
def recommend_movies(user_description, movie_data, top_n=5, sample_size=5000):
    """
    Recommend movies based on a user's description by calculating similarity scores.

    Args:
        user_description (str): The user's movie description query.
        movie_data (pd.DataFrame): The movie dataset containing movie titles and descriptions.
        top_n (int): The number of top recommendations to return.
        sample_size (int): The number of rows to sample for similarity calculation.

    Returns:
        pd.DataFrame: Top N recommended movies.
    """
    # Clean the dataset (handle missing descriptions)
    movie_data = clean_data(movie_data)

    # Limit the dataset size by sampling
    movie_data_sample = limit_dataset_size(movie_data, sample_size)
    
    # Add the user's description to the dataset (with a placeholder title)
    extended_data = movie_data_sample.copy()
    extended_data.loc[len(extended_data)] = ["User Query", user_description, None]  # Adding user description

    # Calculate similarity scores
    similarity_matrix = calculate_similarity(extended_data['description'].tolist())

    # Get similarity scores for the user's description (last row)
    user_similarity_scores = similarity_matrix[-1][:-1]  # Exclude the user's own description

    # Get indices of the top N most similar movies
    top_indices = user_similarity_scores.argsort()[-top_n:][::-1]

    # Return the top N recommended movies
    return extended_data.iloc[top_indices]

# Example usage:
movie_data = pd.read_csv('data/processed/processed_movies.csv')  # Your full movie dataset
user_description = "movie about toys"
top_n_recommendations = recommend_movies(user_description, movie_data, top_n=5)
print(top_n_recommendations)

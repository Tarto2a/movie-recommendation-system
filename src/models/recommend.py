import pandas as pd
from src.models.similarity import calculate_similarity
from src.data.preprocess import preprocess_text  # Add this import

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
        
        movie_data = movie_data.head(sample_size)
    
    return movie_data

def recommend_movies(user_description, movie_data, top_n=5):
    """
    Recommend movies based on a user's description by calculating similarity scores.

    Args:
        user_description (str): The user's movie description query.
        movie_data (pd.DataFrame): The movie dataset containing movie titles and descriptions.
        top_n (int): The number of top recommendations to return.

    Returns:
        pd.DataFrame: Top N recommended movies with non-zero similarity scores.
    """
    import numpy as np

    # Ensure the movie_data has the necessary columns
    if 'title' not in movie_data.columns or 'description' not in movie_data.columns:
        raise ValueError("The movie data must have 'title' and 'description' columns.")
    
    # Handle NaN values in the descriptions by filling them with an empty string or placeholder
    movie_data['description'] = movie_data['description'].fillna('No description available')
    
    processed_user_description = preprocess_text(user_description)

    # Add the user's description as the last entry in the dataset
    extended_data = movie_data.copy()
    extended_data.loc[len(extended_data)] = ["User Query", user_description, processed_user_description]

    # Preprocess the user's description if needed
    # processed_user_description = preprocess_text(user_description)

    # Calculate similarity between the user's description and movie descriptions
    similarity_matrix = calculate_similarity(extended_data['description'].tolist())

    # Get similarity scores for the user's description (last row)
    user_similarity_scores = similarity_matrix[-1][:-1]  # Exclude the user's own description

    # Filter out movies with zero similarity
    non_zero_indices = np.where(user_similarity_scores > 0)[0]
    non_zero_scores = user_similarity_scores[non_zero_indices]

    # If there are no valid recommendations, return an empty DataFrame
    if len(non_zero_scores) == 0:
        return pd.DataFrame(columns=movie_data.columns)

    # Get indices of the top N most similar movies
    top_indices = non_zero_indices[np.argsort(non_zero_scores)[-top_n:][::-1]]

    # Return the top N recommended movies
    return movie_data.iloc[top_indices]

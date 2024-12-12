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
        pd.DataFrame: Top N recommended movies.
    """
    # Ensure the movie_data has the necessary columns
    if 'title' not in movie_data.columns or 'description' not in movie_data.columns:
        raise ValueError("The movie data must have 'title' and 'description' columns.")
    
    # Handle NaN values in the descriptions by filling them with an empty string or placeholder
    movie_data['description'] = movie_data['description'].fillna('No description available')

    # Add the user's description as the last entry in the dataset with a placeholder title
    extended_data = movie_data.copy()

    # Optionally, preprocess the user's description (e.g., lemmatize or remove stopwords)
    processed_user_description = preprocess_text(user_description)  # If you want to apply preprocessing

    # Add the row for the user's query: [title, description, processed_description]
    extended_data.loc[len(extended_data)] = ["User Query", user_description, processed_user_description]  

    # Calculate similarity between the user's description and movie descriptions
    similarity_matrix = calculate_similarity(extended_data['description'].tolist())

    # Get similarity scores for the user's description (last row)
    user_similarity_scores = similarity_matrix[-1][:-1]  # Exclude the user's own description

    # Get indices of the top N most similar movies
    top_indices = user_similarity_scores.argsort()[-top_n:][::-1]

    # Return the top N recommended movies
    return extended_data.iloc[top_indices]

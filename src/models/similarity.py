from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(corpus):
    """
    Calculate the cosine similarity between movie descriptions in the corpus.

    Args:
        corpus (list of str): A list of movie descriptions.

    Returns:
        scipy.sparse.matrix: Matrix of pairwise cosine similarity scores.
    """
    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()

    # Convert descriptions to TF-IDF vectors
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Calculate cosine similarity between each pair of movie descriptions
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return similarity_matrix

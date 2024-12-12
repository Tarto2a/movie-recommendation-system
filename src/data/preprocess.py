import nltk
import pandas as pd
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize stopwords and lemmatizer
STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

def preprocess_text(text):
    """
    Preprocess a given text: tokenization, removing stopwords and punctuation, and lemmatization.

    Args:
        text (str): The text to preprocess.

    Returns:
        str: The cleaned and preprocessed text.
    """
    if not isinstance(text, str):  # Handle non-string values
        return ""
    
    # Lowercase the text
    text = text.lower()

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove punctuation and stopwords, and lemmatize tokens
    tokens = [
        LEMMATIZER.lemmatize(word)
        for word in tokens
        if word not in STOPWORDS and word not in string.punctuation
    ]

    # Join tokens back into a single string
    return " ".join(tokens)

if __name__ == "__main__":
    # Define paths for input and output CSV files
    input_file = os.path.join("data", "raw", "movies.csv")  # Input file path
    output_file = os.path.join("data", "processed", "processed_movies.csv")  # Output file path

    # Check if the input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist.")
    else:
        # Read the input CSV file
        df = pd.read_csv(input_file)

        # Ensure 'description' column exists
        if 'description' not in df.columns:
            print(f"Error: 'description' column not found in '{input_file}'.")
        else:
            # Preprocess the 'description' column
            df['processed_description'] = df['description'].apply(preprocess_text)

            # Save the processed data to the output CSV file
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            df.to_csv(output_file, index=False)

            print(f"Processed data has been saved to '{output_file}'.")

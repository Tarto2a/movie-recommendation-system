import sys
from ..data.loader import load_processed_data  # Use load_processed_data
from ..models.recommend import recommend_movies,limit_dataset_size

def main():
    # Load processed data (movies dataset)
    movie_data = load_processed_data()

    if movie_data is None:
        print("Error: Could not load the processed movie data.")
        sys.exit(1)

    # Get user input for the movie description
    user_description = input("Enter a description of the movie you want to watch: ").strip()

    if not user_description:
        print("Error: Description cannot be empty.")
        sys.exit(1)

    # Get movie recommendations
    recommendations = recommend_movies(user_description, limit_dataset_size(movie_data))

    # Check if recommendations are empty
    if recommendations.empty:
        print("No recommendations found based on your description.")
    else:
        # Display the recommendations
        print("\nTop movie recommendations based on your description:")
        for index, row in recommendations.iterrows():
            print(f"{row['title']} - {row['description']}")

if __name__ == "__main__":
    main()

from flask import Flask, render_template, request
import pandas as pd
from ..data.loader import load_processed_data
from ..models.recommend import recommend_movies, limit_dataset_size

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = []
    if request.method == "POST":
        user_description = request.form["description"]
        movie_data = load_processed_data()

        if movie_data.empty:  # Check if DataFrame is empty
            return "Error loading movie data or data is empty."

        # Limit the dataset size before passing it to the recommendation function
        movie_data = limit_dataset_size(movie_data, sample_size=10000)

        recommendations = recommend_movies(user_description, movie_data)

    # Debugging: print recommendations to see its structure
    print(recommendations)  # Check the structure of recommendations

    if isinstance(recommendations, list) and len(recommendations) > 0:
        return render_template("index.html", recommendations=recommendations)
    elif isinstance(recommendations, pd.DataFrame) and not recommendations.empty:
        return render_template("index.html", recommendations=recommendations.to_dict(orient='records'))
    else:
        return render_template("index.html", recommendations=[])

if __name__ == "__main__":
    app.run(debug=True)

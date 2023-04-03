# Movie Recommender System

This is a project that recommends movies based on a user's selection of a movie from a list. After a user selects a movie, the program recommends five other movies that are similar to the selected movie, along with their posters.

The movie recommendation system uses natural language processing (NLP) and machine learning algorithms to analyze movie plots and metadata, such as genre and director, to find movies that are similar to the selected movie.

## How to Use
To use the movie recommender, follow these steps:

1. Clone this repository to your local machine.
2. Install the necessary dependencies by running pip install -r requirements.txt in your terminal.
3. Run the program by running python movie-recommender-system.py in your terminal.
4. Select a movie from the list of available movies.
5. The program will recommend five similar movies, along with their posters.

## Data
The list of available movies is stored in a CSV file. The posters for each movie are retrieved using the The Movie Database (TMDb) API, which requires an API key. You will need to obtain an API key and set it as an environment variable named TMDB_API_KEY in order to retrieve the movie posters.

## Credits
This project was created by Kunal Madan.

## Link to the dataset
https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

"""
Configuration settings for the Movie Recommendation System.
"""

# Data paths
DATA_PATH = './data/'
MOVIES_FILE = 'movies.csv'
RATINGS_FILE = 'ratings.csv'
CREDITS_FILE = 'credits.csv'

# Recommendation settings
DEFAULT_RECOMMENDATIONS = 10
MIN_VOTES_PERCENTILE = 0.9

# Model settings
SVD_COMPONENTS = 50
TEST_SIZE = 0.2
RANDOM_STATE = 42

# TF-IDF settings
MAX_FEATURES = 50000
OVERVIEW_WEIGHT = 0.6
KEYWORDS_WEIGHT = 0.25
GENRES_WEIGHT = 0.15

# Visualization settings
FIGURE_SIZE = (12, 8)
FONT_SIZE = 12
DPI = 100

# Random seed for reproducibility
RANDOM_SEED = 42

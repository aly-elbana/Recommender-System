#!/usr/bin/env python3
"""
Advanced Movie Recommendation System

This script implements three different recommendation approaches:
1. Popularity-Based Filtering
2. Collaborative Filtering (SVD)
3. Content-Based Filtering (TF-IDF)

Author: Data Science Project
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import ast
import re
import unicodedata
from pathlib import Path

# Machine Learning imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Configure environment
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
sns.set_palette("husl")
np.random.seed(42)

class MovieRecommender:
    """
    Comprehensive movie recommendation system combining multiple approaches.
    """
    
    def __init__(self, data_path='./data/'):
        """Initialize the recommender with data path."""
        self.data_path = Path(data_path)
        self.movies = None
        self.ratings = None
        self.credits = None
        self.svd_model = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        
    def load_data(self):
        """Load all datasets."""
        print("Loading datasets...")
        try:
            self.movies = pd.read_csv(self.data_path / 'movies.csv')
            self.ratings = pd.read_csv(self.data_path / 'ratings.csv')
            self.credits = pd.read_csv(self.data_path / 'credits.csv')
            
            print(f"✓ Movies: {len(self.movies):,} records")
            print(f"✓ Ratings: {len(self.ratings):,} records")
            print(f"✓ Credits: {len(self.credits):,} records")
            return True
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            return False
    
    def popularity_based_recommendations(self, n_recommendations=10, min_votes_percentile=0.9):
        """
        Generate popularity-based recommendations using weighted rating formula.
        
        Args:
            n_recommendations (int): Number of recommendations to return
            min_votes_percentile (float): Percentile for minimum votes threshold
            
        Returns:
            pd.DataFrame: Top movies by weighted rating
        """
        print("\n🎬 Generating Popularity-Based Recommendations...")
        
        # Calculate parameters for weighted rating
        m = self.movies['vote_count'].quantile(min_votes_percentile)
        C = self.movies['vote_average'].mean()
        
        print(f"Minimum votes required: {m:.0f}")
        print(f"Mean vote across dataset: {C:.3f}")
        
        def weighted_rating(df, m=m, C=C):
            """Calculate weighted rating using IMDB formula."""
            R = df['vote_average']
            v = df['vote_count']
            return ((v / (v + m)) * R) + ((m / (v + m)) * C)
        
        # Apply weighted rating
        self.movies['weighted_rating'] = self.movies.apply(weighted_rating, axis=1)
        
        # Get top recommendations
        top_movies = self.movies.sort_values('weighted_rating', ascending=False).head(n_recommendations)
        
        print(f"\nTop {n_recommendations} Popular Movies:")
        for i, (idx, row) in enumerate(top_movies.iterrows(), 1):
            print(f"{i:2d}. {row['title']}")
            print(f"    Rating: {row['vote_average']:.1f} | Votes: {row['vote_count']:,} | Weighted: {row['weighted_rating']:.3f}")
        
        return top_movies[['title', 'vote_average', 'vote_count', 'weighted_rating']]
    
    def collaborative_filtering_recommendations(self, user_id, n_recommendations=10):
        """
        Generate collaborative filtering recommendations using sklearn's TruncatedSVD.
        
        Args:
            user_id (int): User ID for recommendations
            n_recommendations (int): Number of recommendations to return
            
        Returns:
            list: Recommended movie titles
        """
        print(f"\n🤝 Generating Collaborative Filtering Recommendations for User {user_id}...")
        
        try:
            # Create user-item matrix
            user_item_matrix = self.ratings.pivot_table(
                index='userId', 
                columns='movieId', 
                values='rating', 
                fill_value=0
            )
            
            # Get user ratings
            if user_id not in user_item_matrix.index:
                print(f"User {user_id} not found in the dataset.")
                return []
            
            user_ratings = user_item_matrix.loc[user_id]
            
            # Apply SVD to the user-item matrix
            svd = TruncatedSVD(n_components=50, random_state=42)
            user_factors = svd.fit_transform(user_item_matrix)
            item_factors = svd.components_
            
            # Get user factor for the specific user
            user_idx = user_item_matrix.index.get_loc(user_id)
            user_factor = user_factors[user_idx]
            
            # Calculate predicted ratings for all movies
            predicted_ratings = np.dot(user_factor, item_factors)
            
            # Get movies not rated by user (or rated as 0)
            unrated_mask = user_ratings == 0
            unrated_movie_ids = user_item_matrix.columns[unrated_mask]
            unrated_predictions = predicted_ratings[unrated_mask]
            
            # Sort by predicted rating and get top recommendations
            sorted_indices = np.argsort(unrated_predictions)[::-1]
            top_movie_ids = unrated_movie_ids[sorted_indices[:n_recommendations]]
            
            # Get movie titles
            recommendations = self.movies[self.movies['id'].isin(top_movie_ids)]['title'].tolist()
            
            print(f"\nTop {n_recommendations} Collaborative Recommendations:")
            for i, title in enumerate(recommendations, 1):
                print(f"{i:2d}. {title}")
            
            return recommendations
            
        except Exception as e:
            print(f"Error in collaborative filtering: {e}")
            return []
    
    def content_based_recommendations(self, movie_title, n_recommendations=10):
        """
        Generate content-based recommendations using TF-IDF and cosine similarity.
        
        Args:
            movie_title (str): Title of the movie to find similar movies for
            n_recommendations (int): Number of recommendations to return
            
        Returns:
            list: Similar movie titles
        """
        print(f"\n📝 Generating Content-Based Recommendations for '{movie_title}'...")
        
        try:
            # Text preprocessing functions
            def safe_parse_list(x):
                if pd.isna(x) or x == "":
                    return []
                try:
                    items = ast.literal_eval(x) if isinstance(x, str) else x
                    if isinstance(items, list):
                        names = []
                        for it in items:
                            if isinstance(it, dict) and 'name' in it:
                                names.append(it['name'])
                            elif isinstance(it, str):
                                names.append(it)
                        return names
                    return []
                except Exception:
                    return []
            
            def normalize_text(text):
                if pd.isna(text):
                    text = ""
                text = str(text).lower()
                text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
                text = re.sub(r"[^\w\s]", " ", text)
                text = re.sub(r"\s+", " ", text).strip()
                return text
            
            # Build combined features
            movies_copy = self.movies.copy()
            
            # Process keywords and genres
            kw_list = movies_copy['keywords'].apply(safe_parse_list) if 'keywords' in movies_copy.columns else [[]]
            gn_list = movies_copy['genres'].apply(safe_parse_list) if 'genres' in movies_copy.columns else [[]]
            
            movies_copy['_kw_str'] = kw_list.apply(lambda lst: " ".join(normalize_text(x) for x in lst))
            movies_copy['_gn_str'] = gn_list.apply(lambda lst: " ".join(normalize_text(x) for x in lst))
            movies_copy['_ov_str'] = movies_copy['overview'].fillna("").apply(normalize_text) if 'overview' in movies_copy.columns else ""
            
            # Weight the features
            def weight_text(text, weight):
                reps = max(1, int(round(weight*3)))
                return (" " + text) * reps if text else ""
            
            movies_copy['combined_features'] = (
                movies_copy['_ov_str'].apply(lambda t: weight_text(t, 0.6)) + " " +
                movies_copy['_kw_str'].apply(lambda t: weight_text(t, 0.25)) + " " +
                movies_copy['_gn_str'].apply(lambda t: weight_text(t, 0.15))
            ).str.strip()
            
            # Create TF-IDF matrix
            tfidf = TfidfVectorizer(max_features=50000, stop_words='english')
            self.tfidf_matrix = tfidf.fit_transform(movies_copy['combined_features'])
            
            # Calculate cosine similarity
            self.cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)
            
            # Find movie index
            movie_indices = pd.Series(movies_copy.index, index=movies_copy['title']).drop_duplicates()
            if movie_title not in movie_indices:
                print(f"Movie '{movie_title}' not found in dataset.")
                return []
            
            idx = movie_indices[movie_title]
            
            # Get similarity scores
            sim_scores = list(enumerate(self.cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            # Get top similar movies (excluding the movie itself)
            top_indices = [i[0] for i in sim_scores[1:n_recommendations+1]]
            recommendations = movies_copy.iloc[top_indices]['title'].tolist()
            
            print(f"\nTop {n_recommendations} Content-Based Recommendations:")
            for i, title in enumerate(recommendations, 1):
                print(f"{i:2d}. {title}")
            
            return recommendations
            
        except Exception as e:
            print(f"Error in content-based filtering: {e}")
            return []
    
    def hybrid_recommendations(self, user_id, movie_title=None, n_recommendations=10):
        """
        Generate hybrid recommendations combining all approaches.
        
        Args:
            user_id (int): User ID for collaborative filtering
            movie_title (str): Movie title for content-based filtering
            n_recommendations (int): Number of recommendations to return
            
        Returns:
            dict: Combined recommendations from all methods
        """
        print(f"\n🔄 Generating Hybrid Recommendations...")
        
        results = {}
        
        # Popularity-based recommendations
        results['popularity'] = self.popularity_based_recommendations(n_recommendations)
        
        # Collaborative filtering recommendations
        results['collaborative'] = self.collaborative_filtering_recommendations(user_id, n_recommendations)
        
        # Content-based recommendations (if movie title provided)
        if movie_title:
            results['content'] = self.content_based_recommendations(movie_title, n_recommendations)
        
        return results
    
    def analyze_data(self):
        """Perform comprehensive data analysis."""
        print("\n📊 Data Analysis:")
        print("="*50)
        
        # Movies analysis
        print(f"Movies Dataset:")
        print(f"  Total movies: {len(self.movies):,}")
        print(f"  Average rating: {self.movies['vote_average'].mean():.2f}")
        print(f"  Average vote count: {self.movies['vote_count'].mean():.0f}")
        print(f"  Rating range: {self.movies['vote_average'].min():.1f} - {self.movies['vote_average'].max():.1f}")
        
        # Ratings analysis
        print(f"\nRatings Dataset:")
        print(f"  Total ratings: {len(self.ratings):,}")
        print(f"  Unique users: {self.ratings['userId'].nunique():,}")
        print(f"  Unique movies: {self.ratings['movieId'].nunique():,}")
        print(f"  Average rating: {self.ratings['rating'].mean():.2f}")
        
        # Rating distribution
        rating_counts = self.ratings['rating'].value_counts().sort_index()
        print(f"\nRating Distribution:")
        for rating, count in rating_counts.items():
            print(f"  {rating}: {count:,} ({count/len(self.ratings)*100:.1f}%)")
    
    def visualize_data(self):
        """Create visualizations for data analysis."""
        print("\n📈 Creating Visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Rating distribution
        rating_counts = self.ratings['rating'].value_counts().sort_index()
        axes[0,0].bar(rating_counts.index, rating_counts.values, color='skyblue', alpha=0.7)
        axes[0,0].set_title('Rating Distribution', fontweight='bold')
        axes[0,0].set_xlabel('Rating')
        axes[0,0].set_ylabel('Count')
        
        # Ratings per user
        user_ratings = self.ratings.groupby('userId').size()
        axes[0,1].hist(user_ratings, bins=50, color='lightgreen', alpha=0.7)
        axes[0,1].set_title('Ratings per User', fontweight='bold')
        axes[0,1].set_xlabel('Number of Ratings')
        axes[0,1].set_ylabel('Number of Users')
        
        # Ratings per movie
        movie_ratings = self.ratings.groupby('movieId').size()
        axes[1,0].hist(movie_ratings, bins=50, color='salmon', alpha=0.7)
        axes[1,0].set_title('Ratings per Movie', fontweight='bold')
        axes[1,0].set_xlabel('Number of Ratings')
        axes[1,0].set_ylabel('Number of Movies')
        
        # Movie vote averages
        axes[1,1].hist(self.movies['vote_average'], bins=50, color='purple', alpha=0.7)
        axes[1,1].set_title('Movie Vote Averages', fontweight='bold')
        axes[1,1].set_xlabel('Average Rating')
        axes[1,1].set_ylabel('Number of Movies')
        
        plt.tight_layout()
        plt.show()
    
    def evaluate_collaborative_model(self):
        """Evaluate the collaborative filtering model using sklearn."""
        print("\n🔍 Evaluating Collaborative Filtering Model...")
        
        try:
            # Create user-item matrix
            user_item_matrix = self.ratings.pivot_table(
                index='userId', 
                columns='movieId', 
                values='rating', 
                fill_value=0
            )
            
            # Split data for evaluation
            train_data, test_data = train_test_split(
                self.ratings, 
                test_size=0.2, 
                random_state=42
            )
            
            # Create training matrix
            train_matrix = train_data.pivot_table(
                index='userId', 
                columns='movieId', 
                values='rating', 
                fill_value=0
            )
            
            # Apply SVD to training data
            svd = TruncatedSVD(n_components=50, random_state=42)
            user_factors = svd.fit_transform(train_matrix)
            item_factors = svd.components_
            
            # Reconstruct ratings matrix
            reconstructed_matrix = np.dot(user_factors, item_factors)
            
            # Get predictions for test data
            predictions = []
            actual_ratings = []
            
            for _, row in test_data.iterrows():
                user_id = row['userId']
                movie_id = row['movieId']
                actual_rating = row['rating']
                
                if user_id in train_matrix.index and movie_id in train_matrix.columns:
                    user_idx = train_matrix.index.get_loc(user_id)
                    movie_idx = train_matrix.columns.get_loc(movie_id)
                    predicted_rating = reconstructed_matrix[user_idx, movie_idx]
                    
                    predictions.append(predicted_rating)
                    actual_ratings.append(actual_rating)
            
            if predictions:
                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(actual_ratings, predictions))
                mae = mean_absolute_error(actual_ratings, predictions)
                
                print(f"RMSE: {rmse:.3f}")
                print(f"MAE: {mae:.3f}")
                print(f"Evaluated on {len(predictions)} test samples")
            else:
                print("No valid predictions could be made for evaluation.")
            
        except Exception as e:
            print(f"Error in model evaluation: {e}")

def main():
    """Main function to demonstrate the recommendation system."""
    print("🎬 Advanced Movie Recommendation System")
    print("="*50)
    print(f"Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize recommender
    recommender = MovieRecommender()
    
    # Load data
    if not recommender.load_data():
        print("Failed to load data. Please check your data files.")
        return
    
    # Analyze data
    recommender.analyze_data()
    
    # Create visualizations
    recommender.visualize_data()
    
    # Generate different types of recommendations
    print("\n" + "="*50)
    print("RECOMMENDATION EXAMPLES")
    print("="*50)
    
    # Popularity-based recommendations
    popularity_recs = recommender.popularity_based_recommendations(n_recommendations=5)
    
    # Collaborative filtering recommendations
    collaborative_recs = recommender.collaborative_filtering_recommendations(user_id=1, n_recommendations=5)
    
    # Content-based recommendations
    content_recs = recommender.content_based_recommendations("The Dark Knight", n_recommendations=5)
    
    # Hybrid recommendations
    hybrid_recs = recommender.hybrid_recommendations(
        user_id=1, 
        movie_title="The Dark Knight", 
        n_recommendations=5
    )
    
    # Evaluate model
    recommender.evaluate_collaborative_model()
    
    print("\n✅ Recommendation system demonstration completed!")

if __name__ == "__main__":
    main()

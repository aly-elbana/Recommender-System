#!/usr/bin/env python3
"""
Simple script to run movie recommendations with different approaches.
"""

from movie_recommender import MovieRecommender
import sys

def main():
    """Run recommendation examples."""
    print("🎬 Movie Recommendation System - Quick Demo")
    print("="*50)
    
    # Initialize recommender
    recommender = MovieRecommender()
    
    # Load data
    if not recommender.load_data():
        print("❌ Failed to load data. Please check your data files.")
        return
    
    # Get user input for recommendations
    print("\nChoose recommendation type:")
    print("1. Popularity-based (top movies)")
    print("2. Collaborative filtering (for specific user)")
    print("3. Content-based (similar movies)")
    print("4. Hybrid (all approaches)")
    print("5. Data analysis only")
    
    try:
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            # Popularity-based
            n = int(input("Number of recommendations (default 10): ") or "10")
            recommender.popularity_based_recommendations(n_recommendations=n)
            
        elif choice == "2":
            # Collaborative filtering
            user_id = int(input("Enter user ID: "))
            n = int(input("Number of recommendations (default 10): ") or "10")
            recommender.collaborative_filtering_recommendations(user_id, n_recommendations=n)
            
        elif choice == "3":
            # Content-based
            movie_title = input("Enter movie title: ")
            n = int(input("Number of recommendations (default 10): ") or "10")
            recommender.content_based_recommendations(movie_title, n_recommendations=n)
            
        elif choice == "4":
            # Hybrid
            user_id = int(input("Enter user ID: "))
            movie_title = input("Enter movie title for content-based: ")
            n = int(input("Number of recommendations (default 5): ") or "5")
            results = recommender.hybrid_recommendations(user_id, movie_title, n_recommendations=n)
            
        elif choice == "5":
            # Data analysis
            recommender.analyze_data()
            recommender.visualize_data()
            recommender.evaluate_collaborative_model()
            
        else:
            print("Invalid choice. Please run the script again.")
            
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()

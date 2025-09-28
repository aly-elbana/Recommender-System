#!/usr/bin/env python3
"""
Demo script showing how to use the Movie Recommendation System.
"""

from movie_recommender import MovieRecommender

def demo_popularity_based():
    """Demo popularity-based recommendations."""
    print("🎬 DEMO: Popularity-Based Recommendations")
    print("="*50)
    
    recommender = MovieRecommender()
    if not recommender.load_data():
        return
    
    # Get top 5 popular movies
    recommendations = recommender.popularity_based_recommendations(n_recommendations=5)
    return recommendations

def demo_collaborative_filtering():
    """Demo collaborative filtering recommendations."""
    print("\n🤝 DEMO: Collaborative Filtering Recommendations")
    print("="*50)
    
    recommender = MovieRecommender()
    if not recommender.load_data():
        return
    
    # Get recommendations for user 1
    recommendations = recommender.collaborative_filtering_recommendations(user_id=1, n_recommendations=5)
    return recommendations

def demo_content_based():
    """Demo content-based recommendations."""
    print("\n📝 DEMO: Content-Based Recommendations")
    print("="*50)
    
    recommender = MovieRecommender()
    if not recommender.load_data():
        return
    
    # Get movies similar to "The Dark Knight"
    recommendations = recommender.content_based_recommendations("The Dark Knight", n_recommendations=5)
    return recommendations

def demo_hybrid():
    """Demo hybrid recommendations."""
    print("\n🔄 DEMO: Hybrid Recommendations")
    print("="*50)
    
    recommender = MovieRecommender()
    if not recommender.load_data():
        return
    
    # Get hybrid recommendations
    results = recommender.hybrid_recommendations(
        user_id=1, 
        movie_title="The Dark Knight", 
        n_recommendations=3
    )
    return results

def demo_analysis():
    """Demo data analysis and visualization."""
    print("\n📊 DEMO: Data Analysis")
    print("="*50)
    
    recommender = MovieRecommender()
    if not recommender.load_data():
        return
    
    # Perform analysis
    recommender.analyze_data()
    
    # Create visualizations
    recommender.visualize_data()
    
    # Evaluate model
    recommender.evaluate_collaborative_model()

def main():
    """Run all demos."""
    print("🎬 Movie Recommendation System - Demo")
    print("="*60)
    
    try:
        # Run all demos
        demo_popularity_based()
        demo_collaborative_filtering()
        demo_content_based()
        demo_hybrid()
        demo_analysis()
        
        print("\n✅ All demos completed successfully!")
        print("\nTo run individual demos, import and call the demo functions:")
        print("from demo import demo_popularity_based, demo_collaborative_filtering, etc.")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        print("Make sure your data files are in the './data/' directory.")

if __name__ == "__main__":
    main()

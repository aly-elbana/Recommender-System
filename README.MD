# Movie Recommendation System

A comprehensive movie recommendation system implementing three different approaches: Popularity-Based, Collaborative Filtering, and Content-Based filtering.

## 📁 Project Structure

```
movie-recommendation-system/
├── data/
│   ├── movies.csv          # Movie metadata (4,803 movies)
│   ├── ratings.csv         # User ratings (100,000+ ratings)
│   └── credits.csv         # Cast and crew information
├── main1.ipynb            # Popularity-Based Filtering
├── main2.ipynb            # Collaborative Filtering (SVD)
├── main3.ipynb            # Content-Based Filtering (TF-IDF)
├── movie_recommender.py   # Main Python script (all approaches)
├── run_recommendations.py # Interactive recommendation script
├── demo.py                # Demo script with examples
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Jupyter Notebook (optional)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/movie-recommendation-system.git
cd movie-recommendation-system

# Install dependencies
pip install -r requirements.txt
```

### Usage Options

#### Option 1: Python Script (Recommended)

```bash
# Run the main recommendation system
python movie_recommender.py

# Or run the interactive version
python run_recommendations.py

# Or run the demo with examples
python demo.py
```

#### Option 2: Jupyter Notebooks

```bash
# Launch Jupyter Notebook
jupyter notebook

# Open and run:
# - main1.ipynb (Popularity-Based)
# - main2.ipynb (Collaborative Filtering)
# - main3.ipynb (Content-Based)
```

## 📊 Dataset Overview

- **Movies**: 4,803 unique movies with metadata
- **Ratings**: 100,000+ user ratings (1-5 scale)
- **Users**: 671 unique users
- **Features**: Genres, keywords, overview, cast, crew

## 🔬 Methodology

### 1. Popularity-Based Filtering (`main1.ipynb`)

- **Algorithm**: Weighted Rating Formula (IMDB-style)
- **Formula**: `WR = (v/(v+m)) × R + (m/(v+m)) × C`
- **Features**: Statistical analysis, data visualization
- **Output**: Top-rated movies by popularity

### 2. Collaborative Filtering (`main2.ipynb`)

- **Algorithm**: Singular Value Decomposition (SVD)
- **Library**: scikit-learn (TruncatedSVD)
- **Features**: User-item matrix factorization
- **Output**: Personalized recommendations

### 3. Content-Based Filtering (`main3.ipynb`)

- **Algorithm**: TF-IDF Vectorization + Cosine Similarity
- **Features**: Text processing, similarity computation
- **Output**: Similar movies based on content

## 🛠️ Technical Implementation

### Key Libraries

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning utilities and collaborative filtering (TruncatedSVD)
- **matplotlib/seaborn**: Data visualization
- **nltk**: Natural language processing

### Performance Features

- Vectorized operations for speed
- Memory-efficient data processing
- Optimized similarity calculations
- Interactive visualizations

## 📈 Results

### Popularity-Based Results

- Top movies: The Shawshank Redemption, Fight Club, The Dark Knight
- Weighted rating range: 6.0 - 8.1
- Statistical significance testing

### Collaborative Filtering Results

- **Algorithm**: TruncatedSVD (50 components)
- **Matrix Factorization**: User-item matrix decomposition
- **Evaluation**: Train-test split with RMSE and MAE metrics
- **Performance**: Depends on data sparsity and user behavior patterns

### Content-Based Results

- TF-IDF matrix: 4,803 × 24,097 features
- Similarity computation: Linear kernel
- Text preprocessing: Lemmatization and normalization

## 🎯 Usage Examples

### Python Script Usage (Recommended)

```python
from movie_recommender import MovieRecommender

# Initialize recommender
recommender = MovieRecommender()

# Load data
recommender.load_data()

# Popularity-based recommendations
popular_movies = recommender.popularity_based_recommendations(n_recommendations=10)

# Collaborative filtering
user_recs = recommender.collaborative_filtering_recommendations(user_id=1, n_recommendations=10)

# Content-based recommendations
similar_movies = recommender.content_based_recommendations("The Dark Knight", n_recommendations=10)

# Hybrid recommendations (combines all approaches)
hybrid_recs = recommender.hybrid_recommendations(user_id=1, movie_title="The Dark Knight")

# Data analysis and visualization
recommender.analyze_data()
recommender.visualize_data()
```

### Jupyter Notebooks

```python
# Popularity-based recommendations
# Open main1.ipynb and run all cells

# Collaborative filtering
# Open main2.ipynb and run all cells

# Content-based filtering
# Open main3.ipynb and run all cells
```

### Key Features

- **Flexible API**: Use individual methods or combine approaches
- **Interactive Mode**: Run `run_recommendations.py` for guided experience
- **Data Analysis**: Built-in visualization and statistics
- **Model Evaluation**: Cross-validation and performance metrics
- **Configurable**: Easy parameter adjustment via `config.py`

## 📊 Visualizations

- **Rating Distributions**: Histograms and box plots
- **Top Movies**: Horizontal bar charts
- **Similarity Matrices**: Heatmaps
- **User Activity**: Scatter plots and trends
- **Genre Analysis**: Pie charts and bar graphs

## 🔧 Configuration

### Data Paths

- Movies: `./data/movies.csv`
- Ratings: `./data/ratings.csv`
- Credits: `./data/credits.csv`

### Parameters

- **Minimum votes**: 90th percentile
- **TF-IDF features**: 50,000 max
- **Similarity threshold**: Configurable
- **Top-N recommendations**: 10 (default)

## 📝 Features

### Data Processing

- Missing value handling
- Text preprocessing and normalization
- Feature engineering and combination
- Data validation and quality checks

### Machine Learning

- Matrix factorization (SVD)
- Text vectorization (TF-IDF)
- Similarity computation (Cosine)
- Cross-validation and evaluation

### Visualization

- Interactive plots with Plotly
- Statistical charts with Matplotlib
- Data exploration with Seaborn
- Performance metrics visualization

## 🚀 Future Enhancements

- [ ] Hybrid recommendation system
- [ ] Real-time recommendation API
- [ ] Deep learning models
- [ ] User interface development
- [ ] Performance optimization
- [ ] A/B testing framework

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Contact

For questions or suggestions, please open an issue or contact the development team.

---

**Made with ❤️ for the data science community**

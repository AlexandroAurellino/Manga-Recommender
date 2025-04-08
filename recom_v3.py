import json
import logging
import re
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
try:
    from fuzzywuzzy import process
    has_fuzzywuzzy = True
except ImportError:
    has_fuzzywuzzy = False
    print("Note: fuzzywuzzy not installed. For better title matching, install it with: pip install fuzzywuzzy")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MangaRecommender:
    """Manga recommendation system with content-based filtering."""
    
    def __init__(self, 
                 synopsis_weight: float = 0.5,
                 genre_weight: float = 0.3,
                 theme_weight: float = 0.1, 
                 numerical_weight: float = 0.1,
                 svd_components: int = 100,
                 diversity_threshold: float = 0.7,
                 max_tfidf_features: int = 5000):
        """
        Initialize the recommender with configurable parameters.
        
        Args:
            synopsis_weight: Weight given to synopsis features
            genre_weight: Weight given to genre features
            theme_weight: Weight given to theme features
            numerical_weight: Weight given to numerical features (score, members)
            svd_components: Number of components for LSA dimensionality reduction
            diversity_threshold: Threshold for diversity filtering (0-1)
            max_tfidf_features: Maximum number of features for TF-IDF vectorizer
        """
        self.synopsis_weight = synopsis_weight
        self.genre_weight = genre_weight
        self.theme_weight = theme_weight
        self.numerical_weight = numerical_weight
        self.svd_components = svd_components
        self.diversity_threshold = diversity_threshold
        self.max_tfidf_features = max_tfidf_features
        
        # Initialize transformers
        self.tfidf = TfidfVectorizer(stop_words='english', max_features=max_tfidf_features)
        self.svd = TruncatedSVD(n_components=svd_components, random_state=42)
        self.genre_mlb = MultiLabelBinarizer()
        self.theme_mlb = MultiLabelBinarizer()
        self.scaler = MinMaxScaler()
        
        # Data storage
        self.data = None
        self.feature_matrix = None
        self.title_to_index = {}
    
    def load_data(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load manga data from a JSON file.
        
        Args:
            file_path: Path to the JSON data file
            
        Returns:
            List of manga data dictionaries
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            json.JSONDecodeError: If the file isn't valid JSON
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Successfully loaded {len(data)} manga entries from {file_path}")
            return data
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in file: {file_path}")
            raise
    
    def preprocess_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Clean and normalize the manga data.
        
        Args:
            data: Raw manga data
            
        Returns:
            List of cleaned manga data dictionaries
        """
        cleaned_data = []
        for entry in data:
            try:
                # Ensure required fields exist
                if not entry.get('Title'):
                    logger.warning(f"Skipping entry with missing title: {entry}")
                    continue
                
                # Clean and normalize fields
                cleaned_entry = {
                    'Title': entry.get('Title', '').strip(),
                    'Score': self._parse_score(entry.get('Score', 0)),
                    'Members': self._parse_number(entry.get('Members', 0)),
                    'Favourites': self._parse_number(entry.get('Favourites', 0)),
                    'Synopsis': self._clean_text(entry.get('Synopsis', '')),
                    'Genres': [g.lower() for g in entry.get('Genres', [])],
                    'Themes': [t.lower() for t in entry.get('Themes', [])],
                    'Demographic': entry.get('Demographic', 'unknown').lower(),
                    'Image URL': entry.get('Image URL', '').strip()
                }
                cleaned_data.append(cleaned_entry)
            except Exception as e:
                logger.warning(f"Error processing entry: {e}")
                continue
        
        # Build title to index mapping
        self.title_to_index = {item['Title'].lower(): i for i, item in enumerate(cleaned_data)}
        logger.info(f"Preprocessed {len(cleaned_data)} manga entries")
        return cleaned_data
    
    def _parse_score(self, score_value: Union[str, int, float]) -> float:
        """Parse and normalize score values."""
        if isinstance(score_value, (int, float)):
            return float(score_value)
        
        if not score_value:
            return 0.0
            
        score_str = str(score_value).lower().replace('n/a', '0')
        try:
            return float(score_str)
        except ValueError:
            return 0.0
    
    def _parse_number(self, num_value: Union[str, int]) -> int:
        """Parse and normalize numerical values."""
        if isinstance(num_value, int):
            return num_value
            
        if not num_value:
            return 0
            
        try:
            if isinstance(num_value, str):
                # Handle string format with commas
                num_str = num_value.replace(',', '')
                if num_str.lower().startswith('favorites:'):
                    num_str = num_str.lower().replace('favorites:', '').strip()
                return int(num_str) if num_str else 0
            return int(num_value)
        except ValueError:
            return 0
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        # Remove newlines and excessive whitespace
        cleaned = re.sub(r'[\n\r]+', ' ', text)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip()
    
    def create_feature_matrix(self, data: List[Dict[str, Any]]) -> np.ndarray:
        """
        Create a feature matrix using TF-IDF, LSA, and categorical features.
        
        Args:
            data: Cleaned manga data
            
        Returns:
            numpy array of feature vectors
        """
        logger.info("Creating feature matrix...")
        
        # Text features: Synopsis with TF-IDF + LSA
        synopses = [m['Synopsis'] for m in data]
        if not any(synopses):
            logger.warning("No synopsis data found")
            synopsis_reduced = np.zeros((len(data), self.svd_components))
        else:
            logger.info("Processing synopsis with TF-IDF and SVD...")
            synopsis_vectors = self.tfidf.fit_transform(synopses)
            synopsis_reduced = self.svd.fit_transform(synopsis_vectors)
            logger.info(f"SVD explained variance: {sum(self.svd.explained_variance_ratio_):.2f}")
        
        # Categorical features: Genres and Themes with separate MLBs
        logger.info("Processing genres...")
        genre_vectors = self.genre_mlb.fit_transform([m['Genres'] for m in data])
        logger.info(f"Found {len(self.genre_mlb.classes_)} unique genres")
        
        logger.info("Processing themes...")
        theme_vectors = self.theme_mlb.fit_transform([m['Themes'] for m in data])
        logger.info(f"Found {len(self.theme_mlb.classes_)} unique themes")
        
        # Numerical features: Score and Members with normalization
        logger.info("Processing numerical features...")
        numerical_data = np.array([[m['Score'], m['Members']] for m in data])
        numerical_features = self.scaler.fit_transform(numerical_data)
        
        # Combine features with weights
        logger.info("Combining features with weights...")
        try:
            feature_matrix = np.hstack([
                synopsis_reduced * self.synopsis_weight,
                genre_vectors * self.genre_weight,
                theme_vectors * self.theme_weight,
                numerical_features * self.numerical_weight
            ])
            logger.info(f"Created feature matrix with shape {feature_matrix.shape}")
            return feature_matrix
        except ValueError as e:
            logger.error(f"Error combining features: {e}")
            raise
    
    def find_manga_index(self, title: str) -> Optional[int]:
        """
        Find the index of a manga by title with fuzzy matching.
        
        Args:
            title: Title to search for
            
        Returns:
            Index of the manga or None if not found
        """
        # Try exact match first (case-insensitive)
        title_lower = title.lower()
        if title_lower in self.title_to_index:
            return self.title_to_index[title_lower]
        
        # Try fuzzy matching if exact match fails and fuzzywuzzy is available
        if self.data is None or not has_fuzzywuzzy:
            return None
        
        matches = process.extractOne(title, [m['Title'] for m in self.data])
        if matches and matches[1] >= 85:  # 85% similarity threshold
            match_title = matches[0]
            match_idx = next((i for i, m in enumerate(self.data) if m['Title'] == match_title), None)
            logger.info(f"Fuzzy matched '{title}' to '{match_title}' ({matches[1]}% similarity)")
            return match_idx
        
        logger.warning(f"No match found for title '{title}'")
        return None
    
    def recommend_by_title(self, title: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Generate recommendations with diversity constraints.
        
        Args:
            title: Title of the manga to base recommendations on
            top_n: Number of recommendations to return
            
        Returns:
            List of recommended manga dictionaries
        """
        # Fix for the array truth value error - check explicitly
        if self.data is None or self.feature_matrix is None or len(self.feature_matrix) == 0:
            logger.error("No data or feature matrix available. Call fit() first.")
            return []
        
        idx = self.find_manga_index(title)
        if idx is None:
            logger.warning(f"Could not find manga with title similar to '{title}'")
            return []
        
        logger.info(f"Generating recommendations for '{title}' (index {idx})")
        
        # Calculate similarity scores
        query_vector = self.feature_matrix[idx].reshape(1, -1)
        sim_scores = cosine_similarity(query_vector, self.feature_matrix)[0]
        
        # Create list of (index, score) tuples and sort by score
        sim_pairs = [(i, score) for i, score in enumerate(sim_scores) if i != idx]
        sim_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Apply diversity filter
        recommended_indices = []
        recommended_embeddings = []
        
        for candidate_idx, score in sim_pairs:
            if len(recommended_indices) >= top_n:
                break
                
            if not recommended_embeddings:
                # Always add the first recommendation
                recommended_indices.append(candidate_idx)
                recommended_embeddings.append(self.feature_matrix[candidate_idx])
            else:
                # Calculate average similarity to existing recommendations
                candidate_vector = self.feature_matrix[candidate_idx].reshape(1, -1)
                rec_matrix = np.vstack(recommended_embeddings)
                avg_similarity = np.mean(cosine_similarity(candidate_vector, rec_matrix)[0])
                
                if avg_similarity < self.diversity_threshold:
                    recommended_indices.append(candidate_idx)
                    recommended_embeddings.append(self.feature_matrix[candidate_idx])
        
        # Return recommended manga entries
        recommendations = [
            {**self.data[i], 'similarity_score': sim_scores[i]} 
            for i in recommended_indices
        ]
        
        logger.info(f"Found {len(recommendations)} diverse recommendations")
        return recommendations
    
    def evaluate(self, test_size: float = 0.2, top_n: int = 5) -> Dict[str, float]:
        """
        Evaluate the recommender system using precision@k and recall@k.
        
        Args:
            test_size: Proportion of data to use for testing
            top_n: Number of recommendations to consider
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.data is None:
            logger.error("No data available for evaluation")
            return {"precision@k": 0.0, "recall@k": 0.0}
        
        logger.info(f"Evaluating recommender with test_size={test_size}, top_n={top_n}")
        
        # Split data while preserving indices
        indices = list(range(len(self.data)))
        train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=42)
        
        # Extract training and test data
        train_data = [self.data[i] for i in train_idx]
        test_data = [self.data[i] for i in test_idx]
        
        # Extract features for training data
        train_features = self.feature_matrix[train_idx]
        
        # For each test item, fit a temporary model on training data and evaluate
        precision_scores = []
        recall_scores = []
        
        for test_item in test_data:
            # Skip items without genres
            if not test_item['Genres']:
                continue
            
            # Define ground truth: items in training set with genre overlap
            relevant_indices = []
            for i, train_item in enumerate(train_data):
                shared_genres = set(test_item['Genres']).intersection(train_item['Genres'])
                if shared_genres:
                    relevant_indices.append(i)
            
            if not relevant_indices:
                continue
            
            # Create a temporary index mapping for this test item
            temp_index = len(train_data)
            temp_feature_matrix = np.vstack([train_features, self.feature_matrix[test_idx[test_data.index(test_item)]]])
            
            # Calculate similarity
            query_vector = temp_feature_matrix[temp_index].reshape(1, -1)
            sim_scores = cosine_similarity(query_vector, train_features)[0]
            
            # Get top-N recommendations
            recommended_indices = np.argsort(sim_scores)[::-1][:top_n]
            
            # Calculate metrics
            relevant_recommended = len(set(recommended_indices).intersection(relevant_indices))
            precision = relevant_recommended / top_n if top_n > 0 else 0
            recall = relevant_recommended / len(relevant_indices) if relevant_indices else 0
            
            precision_scores.append(precision)
            recall_scores.append(recall)
        
        # Calculate average metrics
        avg_precision = np.mean(precision_scores) if precision_scores else 0.0
        avg_recall = np.mean(recall_scores) if recall_scores else 0.0
        
        logger.info(f"Evaluation results: precision@{top_n}={avg_precision:.4f}, recall@{top_n}={avg_recall:.4f}")
        return {
            "precision@k": avg_precision,
            "recall@k": avg_recall
        }
    
    def fit(self, file_path: str) -> 'MangaRecommender':
        """
        Load data, preprocess, and create feature matrix.
        
        Args:
            file_path: Path to the JSON data file
            
        Returns:
            Self for method chaining
        """
        raw_data = self.load_data(file_path)
        self.data = self.preprocess_data(raw_data)
        self.feature_matrix = self.create_feature_matrix(self.data)
        return self
    
    def get_manga_details(self, title: str) -> Optional[Dict[str, Any]]:
        """Get details for a specific manga by title."""
        idx = self.find_manga_index(title)
        if idx is not None:
            return self.data[idx]
        return None
        
    def get_similar_genres(self, title: str, top_n: int = 5) -> List[str]:
        """Get the most similar genres to those of the given manga."""
        manga = self.get_manga_details(title)
        if not manga or not manga['Genres']:
            return []
            
        # Count genre co-occurrences
        genre_counts = {}
        target_genres = set(manga['Genres'])
        
        for item in self.data:
            if item['Title'] == manga['Title']:
                continue
                
            item_genres = set(item['Genres'])
            overlap = target_genres.intersection(item_genres)
            if overlap:
                for genre in item_genres - target_genres:  # Only count non-overlapping genres
                    genre_counts[genre] = genre_counts.get(genre, 0) + len(overlap)
        
        # Sort by count and return top N
        sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)
        return [g[0] for g in sorted_genres[:top_n]]


def main():
    recommender = MangaRecommender(
        synopsis_weight=0.4,
        genre_weight=0.35, 
        theme_weight=0.15,
        numerical_weight=0.1,
        diversity_threshold=0.65
    )
    
    try:
        # Load data and fit the model
        recommender.fit("cleaned_manga_data.json")
        
        # Evaluate
        metrics = recommender.evaluate()
        print(f"Precision@5: {metrics['precision@k']:.4f}")
        print(f"Recall@5: {metrics['recall@k']:.4f}")
        
        # Generate recommendations
        title = "Berserk"
        recommendations = recommender.recommend_by_title(title)
        
        print(f"\nRecommendations for '{title}':")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['Title']} (Score: {rec['Score']:.2f}, Similarity: {rec['similarity_score']:.4f})")
            genres_display = ', '.join(rec['Genres'][:3])
            if genres_display:
                print(f"   Genres: {genres_display}")
        
        # Show similar genres
        similar_genres = recommender.get_similar_genres(title)
        if similar_genres:
            print(f"\nSimilar genres to {title}: {', '.join(similar_genres)}")
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)


if __name__ == "__main__":
    main()
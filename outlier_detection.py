import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import os
import joblib
from datetime import datetime
from tqdm import tqdm

class OutlierDetector:
    def __init__(self, contamination=0.01, n_estimators=300, n_neighbors=50, random_state=42):
        """
        Initialize the outlier detector.
        
        Args:
            contamination: Expected proportion of outliers
            n_estimators: Number of estimators for Isolation Forest
            n_neighbors: Number of neighbors for LOF
            random_state: Random state for reproducibility
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        
        # Initialize models
        self.iso_forest = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state
        )
        
        self.lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            metric="cosine",
            contamination=contamination,
            novelty=True  # Enable predict method
        )
        
    def fit(self, features):
        """
        Fit outlier detection models.
        
        Args:
            features: Features to fit on
            
        Returns:
            Self
        """
        print("Fitting Isolation Forest...")
        self.iso_forest.fit(features)
        
        print("Fitting Local Outlier Factor...")
        self.lof.fit(features)
        
        return self
    
    def predict_anomaly_scores(self, features, topic_probs=None):
        """
        Predict anomaly scores for features.
        
        Args:
            features: Features to predict on
            topic_probs: Topic probabilities (for novelty score)
            
        Returns:
            DataFrame with anomaly scores
        """
        print("Predicting Isolation Forest scores...")
        # For Isolation Forest, decision_function returns the anomaly score
        # Lower (more negative) scores indicate outliers
        iso_scores = -self.iso_forest.decision_function(features)
        
        print("Predicting Local Outlier Factor scores...")
        # For LOF, decision_function returns the anomaly score
        # Lower (more negative) scores indicate outliers
        lof_scores = -self.lof.decision_function(features)
        
        # Create DataFrame with scores
        scores_df = pd.DataFrame({
            'iso_score': iso_scores,
            'lof_score': lof_scores
        })
        
        # Add novelty score if topic_probs is provided
        if topic_probs is not None:
            print("Calculating novelty scores...")
            novelty_scores = 1 - topic_probs.max(axis=1)
            scores_df['novelty_score'] = novelty_scores
        
        # Normalize scores to [0, 1] range
        for col in scores_df.columns:
            scores_df[col] = (scores_df[col] - scores_df[col].min()) / (scores_df[col].max() - scores_df[col].min())
        
        return scores_df
    
    def combine_scores(self, scores_df, weights=None):
        """
        Combine anomaly scores using weights.
        
        Args:
            scores_df: DataFrame with anomaly scores
            weights: Dictionary of weights for each score
            
        Returns:
            Combined anomaly scores
        """
        if weights is None:
            if 'novelty_score' in scores_df.columns:
                weights = {'iso_score': 0.5, 'lof_score': 0.3, 'novelty_score': 0.2}
            else:
                weights = {'iso_score': 0.7, 'lof_score': 0.3}
        
        print(f"Combining scores with weights: {weights}")
        
        # Combine scores
        combined_score = np.zeros(len(scores_df))
        for col, weight in weights.items():
            combined_score += weight * scores_df[col]
        
        return combined_score
    
    def flag_anomalies(self, combined_score, threshold_percentile=0.99):
        """
        Flag anomalies based on combined score.
        
        Args:
            combined_score: Combined anomaly scores
            threshold_percentile: Percentile threshold for flagging
            
        Returns:
            Binary flags (1 for anomaly, 0 for normal)
        """
        threshold = np.percentile(combined_score, 100 * threshold_percentile)
        print(f"Flagging anomalies with threshold {threshold:.4f} (percentile: {threshold_percentile:.1%})")
        
        flags = (combined_score >= threshold).astype(int)
        n_anomalies = flags.sum()
        
        print(f"Flagged {n_anomalies} anomalies ({n_anomalies / len(flags):.2%} of data)")
        
        return flags
    
    def save_model(self, output_dir='models'):
        """
        Save outlier detection models.
        
        Args:
            output_dir: Directory to save models to
            
        Returns:
            None
        """
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get current date for versioning
        date_str = datetime.now().strftime("%Y%m%d")
        
        # Save Isolation Forest model
        iso_path = os.path.join(output_dir, f'isolation_forest_{date_str}.joblib')
        joblib.dump(self.iso_forest, iso_path)
        
        # Save LOF model
        lof_path = os.path.join(output_dir, f'lof_{date_str}.joblib')
        joblib.dump(self.lof, lof_path)
        
        print(f"Models saved to {output_dir}")
        
    def load_model(self, iso_path, lof_path):
        """
        Load outlier detection models.
        
        Args:
            iso_path: Path to Isolation Forest model
            lof_path: Path to LOF model
            
        Returns:
            None
        """
        self.iso_forest = joblib.load(iso_path)
        self.lof = joblib.load(lof_path)
        print("Models loaded successfully")

def main(reduced_embeddings_file='reduced_embeddings.npz', 
         meta_features_file='meta_features.csv',
         output_file='detected_anomalies.csv',
         contamination=0.01):
    """
    Main function to detect outliers.
    
    Args:
        reduced_embeddings_file: Path to the reduced embeddings file
        meta_features_file: Path to the meta-features file
        output_file: Path to save the detected anomalies to
        contamination: Expected proportion of outliers
    """
    # Load reduced embeddings
    print(f"Loading reduced embeddings from {reduced_embeddings_file}...")
    data = np.load(reduced_embeddings_file)
    reduced_embeddings = data['reduced_embeddings']
    review_ids = data['review_ids']
    
    # Load meta-features
    print(f"Loading meta-features from {meta_features_file}...")
    meta_df = pd.read_csv(meta_features_file)
    
    # Extract topic probabilities
    topic_cols = [col for col in meta_df.columns if col.startswith('topic_prob_')]
    topic_probs = meta_df[topic_cols].values
    
    # Combine reduced embeddings with meta-features
    print("Combining features...")
    meta_features = meta_df[['sentiment_score', 'profanity_flag', 'review_len_log', 'novelty_score']].values
    combined_features = np.hstack([reduced_embeddings, meta_features])
    
    print(f"Combined feature shape: {combined_features.shape}")
    
    # Initialize outlier detector
    detector = OutlierDetector(contamination=contamination)
    
    # Fit detector
    detector.fit(combined_features)
    
    # Predict anomaly scores
    scores_df = detector.predict_anomaly_scores(combined_features, topic_probs)
    
    # Combine scores
    combined_score = detector.combine_scores(scores_df)
    
    # Flag anomalies
    flags = detector.flag_anomalies(combined_score)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'review_id': review_ids,
        'anomaly_score': combined_score,
        'is_anomaly': flags
    })
    
    # Merge with meta_df to get additional information
    results_df = results_df.merge(
        meta_df[['review_id', 'listing_id', 'neighbourhood', 'created_at', 'language', 'rating', 'review_text']],
        on='review_id',
        how='left'
    )
    
    # Save results
    print(f"Saving detected anomalies to {output_file}...")
    results_df.to_csv(output_file, index=False)
    
    # Save models
    detector.save_model()
    
    print("Anomaly detection complete!")
    print(f"Total anomalies detected: {flags.sum()} ({flags.sum() / len(flags):.2%} of data)")

if __name__ == "__main__":
    main() 
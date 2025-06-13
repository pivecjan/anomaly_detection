import pandas as pd
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
from tqdm import tqdm
import os

# Download NLTK resources
nltk.download('vader_lexicon', quiet=True)

# Profanity list (simplified version)
PROFANITY_WORDS = {
    'fuck', 'shit', 'ass', 'bitch', 'damn', 'crap', 'hell', 'bastard', 
    'motherfucker', 'asshole', 'bullshit', 'cunt', 'dick', 'piss', 'slut',
    'whore', 'idiot', 'stupid', 'hate'
}

class MetaFeatureExtractor:
    def __init__(self, n_topics=20):
        """
        Initialize the meta-feature extractor.
        
        Args:
            n_topics: Number of topics for BERTopic
        """
        self.n_topics = n_topics
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.topic_model = None
        
    def extract_sentiment(self, text):
        """
        Extract sentiment score from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment score between -1 and 1
        """
        if not isinstance(text, str) or not text.strip():
            return 0.0
        
        scores = self.sentiment_analyzer.polarity_scores(text)
        return scores['compound']  # Range from -1 (negative) to 1 (positive)
    
    def check_profanity(self, text):
        """
        Check if text contains profanity.
        
        Args:
            text: Text to check
            
        Returns:
            1 if profanity detected, 0 otherwise
        """
        if not isinstance(text, str) or not text.strip():
            return 0
        
        # Convert to lowercase and tokenize
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Check if any word is in the profanity list
        return 1 if any(word in PROFANITY_WORDS for word in words) else 0
    
    def compute_log_length(self, text):
        """
        Compute log of text length.
        
        Args:
            text: Text to analyze
            
        Returns:
            Log of text length
        """
        if not isinstance(text, str) or not text.strip():
            return 0.0
        
        return np.log1p(len(text))
    
    def train_topic_model(self, texts):
        """
        Train BERTopic model on texts.
        
        Args:
            texts: List of texts to train on
            
        Returns:
            Trained BERTopic model
        """
        print("Training BERTopic model...")
        # Configure vectorizer for better topic modeling
        vectorizer_model = CountVectorizer(stop_words="english", 
                                          min_df=5,   # Minimum document frequency
                                          max_df=0.7) # Maximum document frequency
        
        # Initialize and train BERTopic
        self.topic_model = BERTopic(nr_topics=self.n_topics,
                                   vectorizer_model=vectorizer_model,
                                   calculate_probabilities=True)
        
        # Fit the model
        self.topic_model.fit_transform(texts)
        print(f"Trained BERTopic model with {len(self.topic_model.get_topic_info())} topics")
        
        return self.topic_model
    
    def get_topic_distribution(self, texts):
        """
        Get topic distribution for texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Array of topic probabilities for each text
        """
        if self.topic_model is None:
            raise ValueError("Topic model not trained. Call train_topic_model first.")
        
        print("Extracting topic distributions...")
        # Get topic distributions
        _, probs = self.topic_model.transform(texts)
        
        return probs
    
    def extract_all_features(self, df, text_column='cleaned_text'):
        """
        Extract all meta-features from a dataframe.
        
        Args:
            df: Pandas DataFrame containing the texts
            text_column: Column name containing the text to analyze
            
        Returns:
            DataFrame with meta-features added
        """
        print("Extracting meta-features...")
        
        # Extract sentiment scores
        print("Extracting sentiment scores...")
        df['sentiment_score'] = df[text_column].apply(self.extract_sentiment)
        
        # Extract profanity flags
        print("Extracting profanity flags...")
        df['profanity_flag'] = df[text_column].apply(self.check_profanity)
        
        # Compute log length
        print("Computing log lengths...")
        df['review_len_log'] = df[text_column].apply(self.compute_log_length)
        
        # Train topic model and extract topic distributions
        self.train_topic_model(df[text_column].tolist())
        topic_probs = self.get_topic_distribution(df[text_column].tolist())
        
        # Add topic probabilities to dataframe
        for i in range(topic_probs.shape[1]):
            df[f'topic_prob_{i}'] = topic_probs[:, i]
        
        # Calculate novelty score (1 - max topic probability)
        df['novelty_score'] = 1 - topic_probs.max(axis=1)
        
        return df
    
    def save_features(self, df, output_file='meta_features.csv'):
        """
        Save meta-features to a CSV file.
        
        Args:
            df: DataFrame with meta-features
            output_file: File to save to
            
        Returns:
            None
        """
        print(f"Saving meta-features to {output_file}...")
        df.to_csv(output_file, index=False)
        print("Meta-features saved successfully!")

def main(input_file='preprocessed_data.csv', output_file='meta_features.csv'):
    """
    Main function to extract meta-features from preprocessed data.
    
    Args:
        input_file: Path to the preprocessed data CSV
        output_file: Path to save the meta-features to
    """
    # Load preprocessed data
    print(f"Loading preprocessed data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Initialize meta-feature extractor
    extractor = MetaFeatureExtractor(n_topics=20)
    
    # Extract meta-features
    df_with_features = extractor.extract_all_features(df)
    
    # Save features
    extractor.save_features(df_with_features, output_file)
    
    print("Meta-feature extraction complete!")

if __name__ == "__main__":
    main() 
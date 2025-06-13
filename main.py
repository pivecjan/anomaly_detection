import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

# Import modules
from preprocess import preprocess_data
from embedding import EmbeddingPipeline
from meta_features import MetaFeatureExtractor
from dim_reduction import DimensionalityReducer
from outlier_detection import OutlierDetector

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Anomaly Detection Pipeline')
    
    parser.add_argument('--input', type=str, default='anomaly_detection.csv',
                        help='Path to input CSV file')
    parser.add_argument('--output', type=str, default='detected_anomalies.csv',
                        help='Path to output CSV file')
    parser.add_argument('--contamination', type=float, default=0.01,
                        help='Expected proportion of outliers')
    parser.add_argument('--n_components', type=int, default=50,
                        help='Number of components for PCA')
    parser.add_argument('--model', type=str, default='intfloat/multilingual-e5-large-instruct',
                        help='Embedding model to use')
    parser.add_argument('--skip_preprocessing', action='store_true',
                        help='Skip preprocessing step')
    parser.add_argument('--skip_embedding', action='store_true',
                        help='Skip embedding step')
    parser.add_argument('--skip_meta_features', action='store_true',
                        help='Skip meta-feature extraction step')
    parser.add_argument('--skip_dim_reduction', action='store_true',
                        help='Skip dimensionality reduction step')
    
    return parser.parse_args()

def main():
    """Main function to run the anomaly detection pipeline."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Define file paths
    preprocessed_file = 'output/preprocessed_data.csv'
    embeddings_file = 'output/embeddings.npz'
    meta_features_file = 'output/meta_features.csv'
    reduced_embeddings_file = 'output/reduced_embeddings.npz'
    
    # Step 1: Preprocess data
    if not args.skip_preprocessing:
        print("\n" + "="*50)
        print("Step 1: Preprocessing data")
        print("="*50)
        df = preprocess_data(args.input, preprocessed_file)
    else:
        print("\n" + "="*50)
        print("Skipping preprocessing, loading preprocessed data...")
        print("="*50)
        df = pd.read_csv(preprocessed_file)
    
    # Step 2: Generate embeddings
    if not args.skip_embedding:
        print("\n" + "="*50)
        print("Step 2: Generating embeddings")
        print("="*50)
        pipeline = EmbeddingPipeline(model_name=args.model)
        embeddings = pipeline.process_dataframe(df, output_file=embeddings_file)
    else:
        print("\n" + "="*50)
        print("Skipping embedding generation")
        print("="*50)
    
    # Step 3: Extract meta-features
    if not args.skip_meta_features:
        print("\n" + "="*50)
        print("Step 3: Extracting meta-features")
        print("="*50)
        extractor = MetaFeatureExtractor(n_topics=20)
        df_with_features = extractor.extract_all_features(df)
        extractor.save_features(df_with_features, meta_features_file)
    else:
        print("\n" + "="*50)
        print("Skipping meta-feature extraction")
        print("="*50)
    
    # Step 4: Reduce dimensionality
    if not args.skip_dim_reduction:
        print("\n" + "="*50)
        print("Step 4: Reducing dimensionality")
        print("="*50)
        # Load embeddings
        data = np.load(embeddings_file)
        embeddings = data['embeddings']
        review_ids = data['review_ids']
        
        # Initialize dimensionality reducer
        reducer = DimensionalityReducer(n_components=args.n_components)
        
        # Reduce dimensionality
        reduced_embeddings = reducer.fit_transform(embeddings)
        
        # Save reduced embeddings
        np.savez_compressed(reduced_embeddings_file, 
                           reduced_embeddings=reduced_embeddings, 
                           review_ids=review_ids)
        
        # Save models
        reducer.save_model()
    else:
        print("\n" + "="*50)
        print("Skipping dimensionality reduction")
        print("="*50)
    
    # Step 5: Detect outliers
    print("\n" + "="*50)
    print("Step 5: Detecting outliers")
    print("="*50)
    
    # Load reduced embeddings
    data = np.load(reduced_embeddings_file)
    reduced_embeddings = data['reduced_embeddings']
    review_ids = data['review_ids']
    
    # Load meta-features
    meta_df = pd.read_csv(meta_features_file)
    
    # Extract topic probabilities
    topic_cols = [col for col in meta_df.columns if col.startswith('topic_prob_')]
    topic_probs = meta_df[topic_cols].values
    
    # Combine reduced embeddings with meta-features
    meta_features = meta_df[['sentiment_score', 'profanity_flag', 'review_len_log', 'novelty_score']].values
    combined_features = np.hstack([reduced_embeddings, meta_features])
    
    print(f"Combined feature shape: {combined_features.shape}")
    
    # Initialize outlier detector
    detector = OutlierDetector(contamination=args.contamination)
    
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
    print(f"Saving detected anomalies to {args.output}...")
    results_df.to_csv(args.output, index=False)
    
    # Save models
    detector.save_model()
    
    print("\n" + "="*50)
    print("Anomaly detection pipeline complete!")
    print(f"Total anomalies detected: {flags.sum()} ({flags.sum() / len(flags):.2%} of data)")
    print("="*50)
    
    # Print top anomalies
    print("\nTop 5 anomalies:")
    top_anomalies = results_df[results_df['is_anomaly'] == 1].sort_values('anomaly_score', ascending=False).head(5)
    for _, row in top_anomalies.iterrows():
        print(f"\nReview ID: {row['review_id']}")
        print(f"Listing ID: {row['listing_id']}")
        print(f"Neighbourhood: {row['neighbourhood']}")
        print(f"Rating: {row['rating']}")
        print(f"Anomaly Score: {row['anomaly_score']:.4f}")
        print(f"Text: {row['review_text'][:100]}...")

if __name__ == "__main__":
    main() 
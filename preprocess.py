import pandas as pd
import numpy as np
import re
import emoji
import html2text
import nltk
from nltk.tokenize import word_tokenize
from datetime import datetime

# Download NLTK resources
nltk.download('punkt', quiet=True)

def clean_text(text):
    """
    Clean text by:
    1. Converting to lowercase
    2. Removing HTML tags
    3. Converting emojis to text codes
    4. Removing extra whitespace
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags
    h = html2text.HTML2Text()
    h.ignore_links = True
    h.ignore_images = True
    text = h.handle(text)
    
    # Replace <br/> with space
    text = re.sub(r'<br/?>', ' ', text)
    
    # Convert emojis to text codes (keep emoji codes if useful)
    text = emoji.demojize(text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_data(input_file='anomaly_detection.csv', output_file='preprocessed_data.csv'):
    """
    Preprocess the data:
    1. Load the CSV
    2. Clean text
    3. Drop reviews with < 5 tokens
    4. Deduplicate identical texts
    5. Save to new CSV
    """
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Rename columns if needed
    if 'raw_text' in df.columns and 'review_text' not in df.columns:
        df = df.rename(columns={'raw_text': 'review_text'})
    
    print(f"Original dataset shape: {df.shape}")
    
    # Clean text
    print("Cleaning text...")
    df['cleaned_text'] = df['review_text'].apply(clean_text)
    
    # Count tokens
    print("Counting tokens...")
    df['token_count'] = df['cleaned_text'].apply(lambda x: len(word_tokenize(x)) if isinstance(x, str) else 0)
    
    # Drop reviews with < 5 tokens
    print("Dropping reviews with < 5 tokens...")
    df_filtered = df[df['token_count'] >= 5].copy()
    print(f"After token filtering: {df_filtered.shape}")
    
    # Deduplicate identical texts
    print("Deduplicating identical texts...")
    df_filtered = df_filtered.drop_duplicates(subset=['cleaned_text'])
    print(f"After deduplication: {df_filtered.shape}")
    
    # Parse dates
    print("Parsing dates...")
    df_filtered['created_at'] = pd.to_datetime(df_filtered['created_at'])
    
    # Save to CSV
    print(f"Saving preprocessed data to {output_file}...")
    df_filtered.to_csv(output_file, index=False)
    print("Preprocessing complete!")
    
    return df_filtered

if __name__ == "__main__":
    preprocess_data() 
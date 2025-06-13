import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os

class EmbeddingPipeline:
    def __init__(self, model_name="intfloat/multilingual-e5-large-instruct", device=None):
        """
        Initialize the embedding pipeline with the specified model.
        
        Args:
            model_name: Name of the model to use for embeddings
            device: Device to use for inference (None for auto-detection)
        """
        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Using device: {device}")
        self.device = device
        
        # Check if CUDA is available and print GPU info
        if device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            
        # Load model
        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        print("Model loaded successfully")
        
    def generate_embeddings(self, texts, batch_size=32):
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for inference
            
        Returns:
            Numpy array of embeddings
        """
        # Format texts for E5 model if using E5
        if "e5" in self.model.get_modules_str().lower():
            texts = [f"passage: {text}" for text in texts]
        
        # Generate embeddings
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.model.encode(batch, show_progress_bar=False)
            embeddings.append(batch_embeddings)
        
        # Concatenate all embeddings
        embeddings = np.vstack(embeddings)
        
        return embeddings
    
    def process_dataframe(self, df, text_column='cleaned_text', output_file='embeddings.npz'):
        """
        Process a dataframe, generate embeddings for the specified text column,
        and save the embeddings to a file.
        
        Args:
            df: Pandas DataFrame containing the texts
            text_column: Column name containing the text to embed
            output_file: File to save the embeddings to
            
        Returns:
            DataFrame with embeddings added
        """
        print(f"Generating embeddings for {len(df)} texts...")
        embeddings = self.generate_embeddings(df[text_column].tolist())
        
        # Save embeddings to file
        print(f"Saving embeddings to {output_file}...")
        np.savez_compressed(output_file, embeddings=embeddings, review_ids=df['review_id'].values)
        
        return embeddings

def main(input_file='preprocessed_data.csv', output_file='embeddings.npz'):
    """
    Main function to generate embeddings for preprocessed data.
    
    Args:
        input_file: Path to the preprocessed data CSV
        output_file: Path to save the embeddings to
    """
    # Load preprocessed data
    print(f"Loading preprocessed data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Initialize embedding pipeline
    pipeline = EmbeddingPipeline()
    
    # Generate embeddings
    embeddings = pipeline.process_dataframe(df, output_file=output_file)
    
    print(f"Embeddings shape: {embeddings.shape}")
    print("Embedding generation complete!")

if __name__ == "__main__":
    main() 
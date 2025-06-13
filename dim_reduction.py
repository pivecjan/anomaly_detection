import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import joblib
from datetime import datetime

class DimensionalityReducer:
    def __init__(self, n_components=50, random_state=42):
        """
        Initialize the dimensionality reducer.
        
        Args:
            n_components: Number of components for PCA
            random_state: Random state for reproducibility
        """
        self.n_components = n_components
        self.random_state = random_state
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self.scaler = StandardScaler()
        
    def fit_transform(self, embeddings, sample_ratio=0.02):
        """
        Fit PCA on a stratified sample and transform the embeddings.
        
        Args:
            embeddings: Numpy array of embeddings
            sample_ratio: Ratio of data to use for fitting PCA
            
        Returns:
            Reduced embeddings
        """
        print(f"Original embedding shape: {embeddings.shape}")
        
        # Take a sample for fitting PCA
        n_samples = int(embeddings.shape[0] * sample_ratio)
        n_samples = max(n_samples, 10)  # Ensure at least 10 samples
        
        print(f"Taking {n_samples} samples ({sample_ratio:.1%}) for fitting PCA...")
        
        # Simple random sampling for now (could be stratified if we had labels)
        indices = np.random.choice(embeddings.shape[0], n_samples, replace=False)
        sample_embeddings = embeddings[indices]
        
        # Fit scaler and PCA on the sample
        print("Fitting StandardScaler...")
        self.scaler.fit(sample_embeddings)
        
        print("Scaling all embeddings...")
        scaled_embeddings = self.scaler.transform(embeddings)
        
        print(f"Fitting PCA with {self.n_components} components...")
        self.pca.fit(scaled_embeddings[indices])
        
        print(f"Explained variance ratio: {sum(self.pca.explained_variance_ratio_):.4f}")
        
        # Transform all embeddings
        print("Transforming all embeddings...")
        reduced_embeddings = self.pca.transform(scaled_embeddings)
        
        print(f"Reduced embedding shape: {reduced_embeddings.shape}")
        
        return reduced_embeddings
    
    def save_model(self, output_dir='models'):
        """
        Save the PCA and scaler models.
        
        Args:
            output_dir: Directory to save models to
            
        Returns:
            None
        """
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get current date for versioning
        date_str = datetime.now().strftime("%Y%m%d")
        
        # Save PCA model
        pca_path = os.path.join(output_dir, f'pca_{self.n_components}_{date_str}.joblib')
        joblib.dump(self.pca, pca_path)
        
        # Save scaler model
        scaler_path = os.path.join(output_dir, f'scaler_{date_str}.joblib')
        joblib.dump(self.scaler, scaler_path)
        
        print(f"Models saved to {output_dir}")
        
    def load_model(self, pca_path, scaler_path):
        """
        Load PCA and scaler models.
        
        Args:
            pca_path: Path to PCA model
            scaler_path: Path to scaler model
            
        Returns:
            None
        """
        self.pca = joblib.load(pca_path)
        self.scaler = joblib.load(scaler_path)
        print("Models loaded successfully")

def main(embeddings_file='embeddings.npz', output_file='reduced_embeddings.npz', n_components=50):
    """
    Main function to reduce dimensionality of embeddings.
    
    Args:
        embeddings_file: Path to the embeddings file
        output_file: Path to save the reduced embeddings to
        n_components: Number of components for PCA
    """
    # Load embeddings
    print(f"Loading embeddings from {embeddings_file}...")
    data = np.load(embeddings_file)
    embeddings = data['embeddings']
    review_ids = data['review_ids']
    
    # Initialize dimensionality reducer
    reducer = DimensionalityReducer(n_components=n_components)
    
    # Reduce dimensionality
    reduced_embeddings = reducer.fit_transform(embeddings)
    
    # Save reduced embeddings
    print(f"Saving reduced embeddings to {output_file}...")
    np.savez_compressed(output_file, 
                       reduced_embeddings=reduced_embeddings, 
                       review_ids=review_ids)
    
    # Save models
    reducer.save_model()
    
    print("Dimensionality reduction complete!")

if __name__ == "__main__":
    main() 
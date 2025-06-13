# Anomaly Detection Pipeline for Reviews

This project implements an anomaly detection pipeline for identifying unusual reviews within listings, neighborhoods, or cities. The pipeline flags reviews that exhibit context-unusual patterns such as sudden sentiment flips, new topics, or toxicity/spam.

## Features

- **Preprocessing**: Cleans text by lowercasing, removing HTML, converting emojis to text codes, and dropping short reviews.
- **Embedding**: Uses multilingual embedding models (default: `intfloat/multilingual-e5-large-instruct`) to convert text to vector representations.
- **Meta-Feature Engineering**: Extracts sentiment scores, profanity flags, review length, and topic distributions.
- **Dimensionality Reduction**: Applies PCA to reduce embedding dimensions while preserving information.
- **Outlier Detection**: Combines Isolation Forest, Local Outlier Factor, and novelty scores to identify anomalous reviews.

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/anomaly_detection.git
cd anomaly_detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the full pipeline with default settings:

```bash
python main.py --input anomaly_detection.csv --output detected_anomalies.csv
```

### Advanced Usage

Customize the pipeline with additional parameters:

```bash
python main.py --input anomaly_detection.csv \
               --output detected_anomalies.csv \
               --contamination 0.01 \
               --n_components 50 \
               --model "intfloat/multilingual-e5-large-instruct"
```

### Skip Steps

You can skip specific steps if you've already run them:

```bash
python main.py --skip_preprocessing --skip_embedding
```

## Pipeline Steps

1. **Preprocessing**:
   - Lowercase text
   - Remove HTML tags
   - Convert emojis to text codes
   - Drop reviews with < 5 tokens
   - Deduplicate identical texts

2. **Embedding**:
   - Generate 1024-dimensional embeddings using a multilingual model
   - Save embeddings to file

3. **Meta-Feature Engineering**:
   - Extract sentiment scores (-1 to +1)
   - Flag profanity (0/1)
   - Compute log review length
   - Generate topic distribution using BERTopic
   - Calculate novelty score (1 - max topic probability)

4. **Dimensionality Reduction**:
   - Apply PCA to reduce embedding dimensions to 50
   - Fit on a 2% stratified sample
   - Save PCA model for future use

5. **Outlier Detection**:
   - Combine reduced embeddings with meta-features
   - Apply Isolation Forest (weight: 0.5)
   - Apply Local Outlier Factor (weight: 0.3)
   - Use novelty score (weight: 0.2)
   - Flag top 1% as anomalies

## Output

The pipeline outputs a CSV file (`detected_anomalies.csv` by default) containing:

- `review_id`: Unique identifier for the review
- `listing_id`: Identifier for the listing
- `neighbourhood`: Neighborhood of the listing
- `created_at`: Review creation date
- `language`: Language of the review
- `rating`: Rating given in the review
- `review_text`: Original review text
- `anomaly_score`: Combined anomaly score
- `is_anomaly`: Binary flag (1 = anomaly, 0 = normal)

## Model Files

The pipeline saves trained models in the `models/` directory:

- PCA model
- StandardScaler
- Isolation Forest
- Local Outlier Factor

## Intermediate Files

Intermediate files are saved in the `output/` directory:

- `preprocessed_data.csv`: Cleaned and filtered reviews
- `embeddings.npz`: Generated embeddings
- `meta_features.csv`: Extracted meta-features
- `reduced_embeddings.npz`: Dimensionality-reduced embeddings

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
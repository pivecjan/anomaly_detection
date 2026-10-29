# Test anomaly detection on feedback

See how the feedback detector surfaces reviews that stand out from the rest. Run the pipeline on a set of reviews, inspect the highest-scoring results, and decide whether the flagged feedback is genuinely unusual.

## What this test does

The pipeline analyzes review text and related features, including sentiment, text length, profanity flags, and topic novelty. It combines Isolation Forest and Local Outlier Factor scores with topic novelty, then writes a score and anomaly flag for each review.

An anomaly is a review that differs from patterns in the data provided. A flag is a prompt for investigation, not proof that a review is fake, abusive, or otherwise incorrect.

## Run a test

1. Install the project dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Prepare a CSV with these columns: `review_id`, `listing_id`, `neighbourhood`, `created_at`, `language`, review text, and `rating`. Review text may be named `raw_text` or `review_text`; if your rating column is named `rating_stars`, rename it to `rating`.

3. Run the detector:

   ```bash
   python main.py --input anomaly_detection.csv --output detected_anomalies.csv
   ```

   On the first run, the embedding model may need to download. Use `--model` to specify another compatible model; run `python main.py --help` for parameter details.

## Review the results

Open `detected_anomalies.csv` and sort by `anomaly_score` from highest to lowest. The file includes the review details, an anomaly score, and `is_anomaly` (`1` for flagged, `0` otherwise). By default, the pipeline flags approximately the top 1% of scores; changing `--contamination` affects the underlying models, but the final flagging threshold is still set to the top 1%.

Check flagged reviews in context and compare them with unflagged feedback. Record whether each flag is useful, then adjust the data or detection approach before relying on results in a live workflow. This project provides an experimental detection pipeline; it does not yet include a feedback review interface or validated accuracy measurements.

#!/bin/bash

# Anomaly Detection Pipeline Runner

# Create output directory if it doesn't exist
mkdir -p output
mkdir -p models

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not found. Please install Python 3."
    exit 1
fi

# Check if input file exists
if [ ! -f "anomaly_detection.csv" ]; then
    echo "Error: Input file 'anomaly_detection.csv' not found."
    exit 1
fi

# Install dependencies if needed
if [ "$1" == "--install" ] || [ "$1" == "-i" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    shift
fi

# Run the pipeline
echo "Running anomaly detection pipeline..."
python main.py "$@"

# Check if the pipeline was successful
if [ $? -eq 0 ]; then
    echo "Pipeline completed successfully!"
    echo "Results saved to detected_anomalies.csv"
else
    echo "Pipeline failed. Check the error messages above."
    exit 1
fi 
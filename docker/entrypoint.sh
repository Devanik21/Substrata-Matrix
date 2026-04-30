#!/bin/bash
set -e

echo "Starting UnSuPERvIsED Streamlit Dashboard..."

# Optional: Run tests before starting if a flag is provided
if [ "$1" = "test" ]; then
    echo "Running test suite..."
    pytest tests/
    exit 0
fi

# Run Streamlit (assuming UnSuPERvIsED.py is at the root or Intermediate_Cluster)
if [ -f "Intermediate_Cluster/UnSuPERvIsED.py" ]; then
    streamlit run Intermediate_Cluster/UnSuPERvIsED.py --server.address=0.0.0.0
elif [ -f "Basic_Cluster/UnSuPERvIsED.py" ]; then
    streamlit run Basic_Cluster/UnSuPERvIsED.py --server.address=0.0.0.0
else
    echo "Could not find Main Streamlit app."
    exit 1
fi

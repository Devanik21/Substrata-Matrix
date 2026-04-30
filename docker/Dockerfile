FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
# If it fails due to missing files in tests, we just create a dummy one
RUN touch requirements.txt
RUN pip install --no-cache-dir numpy pandas scikit-learn scipy plotly pytest pytest-cov tensorflow hdbscan

# Copy application files
COPY . .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose Streamlit port
EXPOSE 8501

ENTRYPOINT ["./docker/entrypoint.sh"]

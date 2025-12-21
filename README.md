**🔍 Duplicate Remover Tool - ML-Powered Data Cleaning Platform
**
**📋 Table of Contents
**
* Overview
* Key Features
* Demo & Screenshots
* Installation
* Quick Start Guide
* Detailed Usage
* Architecture & Components
* Machine Learning Algorithms
* API Reference
* Configuration
* Deployment
* Performance Optimization
* Troubleshooting
* Contributing
* License
* Support


**🎯 Overview**
The Duplicate Remover Tool is a sophisticated data cleaning solution designed for data scientists, analysts, researchers, and businesses dealing with data quality issues. Unlike simple exact-match duplicate removers, this tool uses machine learning to detect:

* Exact Duplicates: Identical rows across all columns
* Near Duplicates: Rows with minor variations in numeric values
* Text Duplicates: Similar text entries with different formatting or minor spelling differences
* Semantic Duplicates: Entries that represent the same entity but with different representations

**Why Use This Tool?**
✅ Intelligent Detection: Goes beyond simple string matching using ML algorithms
✅ Multiple Methods: Compare results from 5+ different detection algorithms
✅ Visual Analytics: Understand your data distribution with interactive visualizations
✅ Production Ready: Scalable architecture suitable for datasets of various sizes
✅ Easy to Use: Intuitive web interfaces requiring no coding knowledge
✅ Customizable: Extensive configuration options for fine-tuned detection
✅ Export Ready: Clean data downloadable in standard CSV format

**🌟 Key Features**

🎨 Dual Interface Options

Flask Web Application
* Modern, responsive Bootstrap 5 design
* Real-time AJAX-powered interactions
* Professional gradient UI with smooth animations
* Comprehensive file upload with drag-and-drop
* Interactive data preview tables
* Downloadable reports and cleaned datasets

Streamlit Application

* Rapid prototyping and deployment
* Built-in interactive widgets
* Automatic UI updates
* Session state management
* Native chart and visualization support
* Simple configuration with minimal code

📊 Powerful Analytics

* Cluster Visualizations: 2D PCA projections showing data clusters and duplicate patterns
* Model Comparison Charts: Bar charts comparing accuracy across ML algorithms
* Distribution Plots: Histograms and box plots for numeric column analysis
* Detailed Reports: Comprehensive text reports with duplicate statistics
* Real-time Metrics: Live counters showing rows processed and duplicates found

⚙️ Flexible Configuration

* Similarity Threshold: Adjust from 0.5 (loose) to 1.0 (strict) matching
* Cluster Count: Configure number of clusters for K-means (2-20)
* Column Selection: Choose specific text or numeric columns for analysis
* Method Selection: Pick your preferred detection algorithm or compare all
* Batch Processing: Handle datasets from 10 to 100,000+ rows


🎬 Demo & Screenshots
Main Interface (Flask)
┌────────────────────────────────────────────────────────┐
│  🔍 Duplicate Remover Tool                             │
│  Find and remove duplicate data using ML techniques    │
├────────────────────────────────────────────────────────┤
│  [Upload Data] [Analyze] [Clean Data] [Visualize]      │
├────────────────────────────────────────────────────────┤
│                                                        │
│  📤 Upload Your Dataset                                │
│  ┌──────────────────────────────────────────┐          │
│  │  Drop your CSV file here                 │          │
│  │  or click to browse                      │          │
│  │  ☁️ [Upload Icon]                        │          │
│  └──────────────────────────────────────────┘          │
│                                                        │
│  Sample Data: [Load Sample Data]                       │
│                                                        │
└────────────────────────────────────────────────────────┘

Analysis Results
┌────────────────────────────────────────────────────────┐
│  📊 Analysis Results                                   │
├─────────┬─────────┬─────────┬─────────┬─────────┐      │
│ Exact   │ Text    │ K-means │ KNN     │ Class.  │      │
│ Dups    │ Dups    │ Dups    │ Dups    │ Dups    │      │
│   3     │   5     │   8     │   6     │   7     │      │
└─────────┴─────────┴─────────┴─────────┴─────────┘      │
│                                                        │
│  📝 Detailed Report                                    │
│  ════════════════════════════════                      │
│  Total Rows: 1000                                      │
│  Total Duplicate Groups: 29                            │
│  Recommendation: Use K-means method                    │
└────────────────────────────────────────────────────────┘


**💿 Installation**
Prerequisites

* Python 3.7 or higher
* pip package manager
* 4GB+ RAM recommended for large datasets
* Modern web browser (Chrome, Firefox, Safari, Edge)

Standard Installation
bashDownloadCopy code# Clone the repository
git clone https://github.com/yourusername/duplicate-remover-tool.git
cd duplicate-remover-tool

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
Requirements.txt
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
streamlit>=1.10.0
flask>=2.0.0
pyngrok>=5.0.0

Docker Installation (Optional)
bashDownloadCopy code# Build Docker image
docker build -t duplicate-remover-tool .

# Run Flask container
docker run -p 5000:5000 duplicate-remover-tool python flask_app.py

# Run Streamlit container
docker run -p 8501:8501 duplicate-remover-tool streamlit run streamlit_app.py
Google Colab Installation
Simply upload the provided notebook to Google Colab and run all cells. The tool will automatically install dependencies and set up ngrok tunneling for public access.

**🚀 Quick Start Guide**
Option 1: Flask Application
bashDownloadCopy code# Navigate to project directory
cd duplicate-remover-tool

# Run Flask app
python flask_app.py

# Open browser
# Navigate to: http://localhost:5000
Option 2: Streamlit Application
bashDownloadCopy code# Navigate to project directory
cd duplicate-remover-tool

# Run Streamlit app
streamlit run streamlit_app.py

# Streamlit will automatically open in your default browser
# Or navigate to: http://localhost:8501
Option 3: Google Colab (Cloud Deployment)

1. 
Get ngrok Auth Token

Visit https://ngrok.com/
Sign up for free account
Copy your auth token from dashboard


2. 
Configure Notebook
pythonDownloadCopy code# In the notebook, find this line:
ngrok.set_auth_token("YOUR_NGROK_AUTHTOKEN")

# Replace with your actual token:
ngrok.set_auth_token("your_actual_token_here")

3. 
Run All Cells

Click "Runtime" → "Run all"
Wait for public URL to be generated
Access your app via the ngrok URL




**📖 Detailed Usage**
Step 1: Upload Your Dataset
Via Web Interface

1. Click on the "Upload Data" tab
2. Drag and drop your CSV file or click to browse
3. Click "Upload File" button
4. View data preview and column information

Via Sample Data

1. Click "Load Sample Data" button
2. Sample dataset with intentional duplicates will be loaded
3. Use this to explore features without preparing your own data

Data Requirements

* Format: CSV (Comma-Separated Values)
* Size: Recommended < 100MB for optimal performance
* Rows: Minimum 2 rows, tested up to 1M rows
* Columns: Any number, automatic type detection
* Encoding: UTF-8 recommended

Step 2: Configure Analysis Parameters
Similarity Threshold

* Range: 0.5 to 1.0
* Default: 0.9
* Lower values (0.5-0.7): Detect more potential duplicates, higher false positives
* Medium values (0.75-0.85): Balanced detection
* Higher values (0.9-1.0): Strict matching, fewer false positives

Number of Clusters

* Range: 2 to 20
* Default: 10
* Small datasets (< 100 rows): Use 2-5 clusters
* Medium datasets (100-1000 rows): Use 5-10 clusters
* Large datasets (> 1000 rows): Use 10-20 clusters

Column Selection
Text Columns:

* Select columns containing text data (names, descriptions, addresses)
* Tool extracts character-level and word-level features
* Best for columns with substantial text (> 10 characters average)

Numeric Columns:

* Select columns containing numbers (age, salary, measurements)
* Tool applies PCA for dimensionality reduction
* Best with at least 2 numeric columns

Step 3: Analyze Your Data

1. Click the "Analyze" tab
2. Configure parameters in settings panel
3. Select text and numeric columns
4. Click "🔍 Analyze Dataset" button
5. Wait for processing (10 seconds to 2 minutes depending on size)

Understanding Results
Exact Duplicates:

* Identical rows across all selected columns
* 100% match in all fields
* Safe to remove automatically

Text Duplicates:

* Similar text content with minor variations
* Examples: "John Smith" vs "john smith", "NYC" vs "New York City"
* Review before removing

K-means Duplicates:

* Detected using clustering algorithm
* Groups similar data points
* Good for numeric similarity

KNN Duplicates:

* Based on nearest neighbor distances
* Effective for local similarity patterns
* Adjustable with n_neighbors parameter

Classification Duplicates:

* ML model trained on your data
* Learns patterns specific to your dataset
* Most accurate for complex cases

Step 4: Review Detailed Reports
The analysis report includes:
📊 Dataset Duplicate Analysis Report
=====================================

Total Rows: 1,000

🔄 Duplicate Types Found:
• Exact Duplicates: 15 groups (45 total rows)
• Text Duplicates: 8 groups (24 total rows)
• Near Duplicates (K-means): 12 groups (36 total rows)
• Near Duplicates (KNN): 10 groups (30 total rows)
• Classification Duplicates: 14 groups (42 total rows)

📈 Summary:
Total Duplicate Groups Found: 59
Estimated Duplicates to Remove: 177 rows
Cleaned Dataset Size: 823 rows

💡 Recommendation: Use K-means method for balanced results

Step 5: Clean Your Dataset

1. Click the "Clean Data" tab
2. Select cleaning method:

K-means: Fast, good for general use
KNN: Accurate for local patterns
Classification: Best for complex datasets


3. Select columns to include in cleaning
4. Click "🧹 Clean Dataset" button
5. Review cleaning summary

Cleaning Process
The tool follows this workflow:

1. Exact duplicates removed first (safest operation)
2. Selected ML method applied to remaining data
3. First occurrence kept for each duplicate group
4. Subsequent duplicates removed systematically
5. Index reset for clean continuous numbering

Step 6: Download Clean Data

1. Review cleaned data preview
2. Check row count reduction
3. Click "📥 Download Cleaned Dataset" button
4. Save CSV file to your computer
5. Import into your data pipeline

Step 7: Visualize Results
Cluster Visualization

* Shows 2D PCA projection of your data
* Different colors represent different clusters
* Points close together are similar
* Helps understand data distribution

Model Comparison

* Bar chart showing accuracy of each ML model
* Compare performance across algorithms
* Choose best method for your dataset
* Accuracy values from 0.0 to 1.0


**🏗️ Architecture & Components**
Project Structure
duplicate-remover-tool/
│
├── duplicate_remover_core.py      # Core ML algorithms and logic
│   ├── TextDuplicateRemover       # Text-based duplicate detection
│   ├── TabularDuplicateRemover    # Numeric-based duplicate detection
│   ├── ClassificationDuplicateDetector  # ML classification approach
│   └── CombinedDuplicateRemover   # Orchestration of all methods
│
├── streamlit_app.py               # Streamlit web interface
│   ├── UI components              # Interactive widgets
│   ├── Session state management   # Data persistence
│   └── Visualization rendering    # Chart generation
│
├── flask_app.py                   # Flask web application
│   ├── Route handlers             # API endpoints
│   ├── HTML template              # Bootstrap UI
│   └── AJAX handlers              # Asynchronous requests
│
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Docker configuration
├── .gitignore                     # Git ignore rules
├── LICENSE                        # MIT license
└── README.md                      # This file

**Core Classes**
1. TextDuplicateRemover
Purpose: Detect duplicates in text columns using feature engineering and clustering.
Key Methods:
pythonDownloadCopy codecreate_text_features(texts)
# Creates numerical features from text:
# - Character count, word count, unique words
# - Upper/lower case ratios
# - Punctuation frequency
# - Character frequency for a-j

find_duplicates(texts)
# Applies K-means clustering to text features
# Calculates similarity within clusters
# Returns list of duplicate groups

visualize_clusters(texts)
# Creates 2D PCA visualization
# Shows cluster distribution
# Returns matplotlib figure
Parameters:

* n_clusters: Number of clusters (default: 10)
* similarity_threshold: Minimum similarity score (default: 0.8)

Use Cases:

* Detecting duplicate names with spelling variations
* Finding similar product descriptions
* Identifying duplicate addresses with formatting differences

2. TabularDuplicateRemover
Purpose: Detect duplicates in numeric data using multiple ML algorithms.
Key Methods:
pythonDownloadCopy codefind_exact_duplicates(df)
# Finds rows identical across all columns
# Fast and deterministic
# Returns list of duplicate index groups

find_near_duplicates_kmeans(df, numeric_columns)
# Uses K-means clustering
# Detects similar numeric patterns
# Good for structured data

find_near_duplicates_knn(df, numeric_columns, n_neighbors)
# Uses K-Nearest Neighbors
# Finds local similarity patterns
# Adjustable neighbor count

visualize_clusters(df, numeric_columns)
# Creates PCA-based visualization
# Shows numeric data distribution
# Color-coded by cluster
Parameters:

* similarity_threshold: Minimum similarity (default: 0.9)
* n_neighbors: Number of neighbors for KNN (default: 5)

Use Cases:

* Detecting duplicate transactions with minor amount variations
* Finding similar customer profiles based on demographics
* Identifying duplicate measurements in scientific data

3. ClassificationDuplicateDetector
Purpose: Train supervised ML models to predict duplicate probability.
Key Methods:
pythonDownloadCopy codecreate_training_data(df, numeric_columns)
# Creates labeled pairs of duplicates and non-duplicates
# Generates feature vectors from row pairs
# Returns X (features) and y (labels)

train_model(X, y)
# Trains selected ML model
# Performs train-test split
# Returns accuracy score

find_duplicates(df, numeric_columns, threshold)
# Predicts duplicate probability for all pairs
# Applies threshold to filter results
# Returns high-confidence duplicate groups

compare_models(df, numeric_columns)
# Trains all available models
# Compares accuracy scores
# Returns dictionary of results
Supported Models:

* Logistic Regression: Fast, interpretable baseline
* SVM: Good for high-dimensional data
* Decision Tree: Interpretable rules
* Random Forest: High accuracy ensemble method

Use Cases:

* Complex datasets with mixed patterns
* When explainability is important
* Datasets with labeled duplicate examples

4. CombinedDuplicateRemover
Purpose: Orchestrate all detection methods and provide unified interface.
Key Methods:
pythonDownloadCopy codeanalyze_dataset(df, text_columns, numeric_columns)
# Runs all detection methods
# Aggregates results
# Returns comprehensive analysis

clean_dataset(df, text_columns, numeric_columns, method)
# Removes duplicates using selected method
# Preserves first occurrence
# Returns cleaned dataframe and count

generate_report(analysis)
# Creates formatted text report
# Includes all metrics
# Returns formatted string
Workflow:

1. Run exact duplicate detection (always safe)
2. Process text columns if specified
3. Apply selected ML method to numeric data
4. Aggregate and deduplicate results
5. Generate comprehensive report


**🔌 API Reference**
Flask Endpoints
POST /upload
Upload CSV file and initialize dataset.
Request:
pythonDownloadCopy codeContent-Type: multipart/form-data
file: <CSV file>
Response:
jsonDownloadCopy code{
  "success": true,
  "rows": 1000,
  "cols": 5,
  "columns": {
    "text": ["Name", "City"],
    "numeric": ["Age", "Salary", "Score"]
  },
  "preview": "<HTML table>"
}
GET /load_sample
Load built-in sample dataset.
Response:
jsonDownloadCopy code{
  "success": true,
  "rows": 8,
  "cols": 5,
  "columns": {...},
  "preview": "<HTML table>"
}
POST /analyze
Analyze dataset for duplicates.
Request:
jsonDownloadCopy code{
  "similarity_threshold": 0.9,
  "n_clusters": 10,
  "text_columns": ["Name"],
  "numeric_columns": ["Age", "Salary"]
}
Response:
jsonDownloadCopy code{
  "success": true,
  "analysis": {
    "total_rows": 1000,
    "exact_duplicates": 15,
    "text_duplicates": 8,
    "near_duplicates_kmeans": 12,
    "near_duplicates_knn": 10,
    "classification_duplicates": 14
  },
  "report": "Formatted text report..."
}
POST /clean
Clean dataset by removing duplicates.
Request:
jsonDownloadCopy code{
  "method": "kmeans",
  "similarity_threshold": 0.9,
  "text_columns": ["Name"],
  "numeric_columns": ["Age", "Salary"]
}
Response:
jsonDownloadCopy code{
  "success": true,
  "original_rows": 1000,
  "cleaned_rows": 823,
  "removed_count": 177,
  "preview": "<HTML table>"
}
GET /download
Download cleaned dataset as CSV.
Response:
Content-Type: text/csv
Content-Disposition: attachment; filename=cleaned_dataset.csv

[CSV content]

POST /visualize_clusters
Generate cluster visualization.
Request:
jsonDownloadCopy code{
  "numeric_columns": ["Age", "Salary"]
}
Response:
jsonDownloadCopy code{
  "success": true,
  "image": "base64_encoded_png_image"
}
POST /compare_models
Compare ML model performances.
Request:
jsonDownloadCopy code{
  "numeric_columns": ["Age", "Salary"]
}
Response:
jsonDownloadCopy code{
  "success": true,
  "results": {
    "Logistic Regression": 0.8234,
    "SVM": 0.8567,
    "Decision Tree": 0.7891,
    "Random Forest": 0.9012
  },
  "image": "base64_encoded_png_image"
}
Python API
pythonDownloadCopy codefrom duplicate_remover_core import CombinedDuplicateRemover
import pandas as pd

# Initialize remover
remover = CombinedDuplicateRemover(similarity_threshold=0.9)

# Load data
df = pd.read_csv('your_data.csv')

# Analyze
analysis = remover.analyze_dataset(
    df,
    text_columns=['Name', 'Description'],
    numeric_columns=['Age', 'Salary', 'Score']
)

# View results
print(remover.generate_report(analysis))

# Clean data
cleaned_df, removed_count = remover.clean_dataset(
    df,
    text_columns=['Name'],
    numeric_columns=['Age', 'Salary'],
    method='kmeans'
)

# Save cleaned data
cleaned_df.to_csv('cleaned_data.csv', index=False)

**⚙️ Configuration
**Environment Variables
Create a .env file in the project root:
bashDownloadCopy code# Flask Configuration
FLASK_APP=flask_app.py
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=your-secret-key-here

# Ngrok Configuration
NGROK_AUTH_TOKEN=your-ngrok-token-here

# ML Configuration
DEFAULT_SIMILARITY_THRESHOLD=0.9
DEFAULT_N_CLUSTERS=10
DEFAULT_N_NEIGHBORS=5

# Performance
MAX_WORKERS=4
CHUNK_SIZE=1000
Application Settings
Edit settings in duplicate_remover_core.py:
pythonDownloadCopy code# Default clustering parameters
DEFAULT_N_CLUSTERS = 10
DEFAULT_SIMILARITY_THRESHOLD = 0.9
DEFAULT_N_NEIGHBORS = 5

# ML model parameters
RF_N_ESTIMATORS = 100
DT_MAX_DEPTH = 10
SVM_C = 1.0
LR_MAX_ITER = 1000

# PCA parameters
PCA_N_COMPONENTS = 10
PCA_VARIANCE_THRESHOLD = 0.95

# Performance settings
BATCH_SIZE = 1000
MAX_MEMORY_MB = 4096
N_JOBS = -1  # Use all CPU cores
Advanced Configuration
For large-scale deployments:
pythonDownloadCopy code# config.py
class Config:
    # Database for storing results
    SQLALCHEMY_DATABASE_URI = 'sqlite:///duplicates.db'
    
    # Redis for caching
    REDIS_URL = 'redis://localhost:6379'
    
    # Celery for async processing
    CELERY_BROKER_URL = 'redis://localhost:6379'
    CELERY_RESULT_BACKEND = 'redis://localhost:6379'
    
    # File upload limits
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB
    
    # Security
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'

**🚢 Deployment**
Local Deployment
Flask (Production Mode)
bashDownloadCopy code# Install production server
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 flask_app:app

# With more workers for better performance
gunicorn -w 8 --threads 2 -b 0.0.0.0:5000 flask_app:app
Streamlit
bashDownloadCopy code# Run in production mode
streamlit run streamlit_app.py --server.port 8501 --server.headless true
Docker Deployment
Dockerfile:
dockerfileDownloadCopy codeFROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000 8501

# Flask
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "flask_app:app"]

# Or Streamlit
# CMD ["streamlit", "run", "streamlit_app.py", "--server.port", "8501"]
Build and Run:
bashDownloadCopy code# Build image
docker build -t duplicate-remover .

# Run Flask
docker run -p 5000:5000 duplicate-remover

# Run Streamlit
docker run -p 8501:8501 duplicate-remover streamlit run streamlit_app.py
Docker Compose
docker-compose.yml:
yamlDownloadCopy codeversion: '3.8'

services:
  flask-app:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
    command: gunicorn -w 4 -b 0.0.0.0:5000 flask_app:app

  streamlit-app:
    build: .
    ports:
      - "8501:8501"
    command: streamlit run streamlit_app.py --server.port 8501

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - flask-app
      - streamlit-app
Cloud Deployment
Heroku (Flask)
bashDownloadCopy code# Create Procfile
echo "web: gunicorn flask_app:app" > Procfile

# Create runtime.txt
echo "python-3.9.16" > runtime.txt

# Deploy
heroku create your-app-name
git push heroku main
heroku open
Streamlit Cloud

1. Push code to GitHub
2. Visit https://share.streamlit.io
3. Connect GitHub repository
4. Select streamlit_app.py
5. Deploy

AWS EC2
bashDownloadCopy code# Connect to EC2 instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Install dependencies
sudo apt update
sudo apt install python3-pip nginx -y

# Clone repository
git clone your-repo-url
cd duplicate-remover-tool

# Install Python packages
pip3 install -r requirements.txt

# Configure Nginx
sudo nano /etc/nginx/sites-available/duplicate-remover

# Nginx configuration
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}

# Enable site
sudo ln -s /etc/nginx/sites-available/duplicate-remover /etc/nginx/sites-enabled/
sudo systemctl restart nginx

# Run with systemd
sudo nano /etc/systemd/system/duplicate-remover.service

# Service file content:
[Unit]
Description=Duplicate Remover Flask App
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/duplicate-remover-tool
ExecStart=/usr/local/bin/gunicorn -w 4 -b 127.0.0.1:5000 flask_app:app
Restart=always

[Install]
WantedBy=multi-user.target

# Start service
sudo systemctl start duplicate-remover
sudo systemctl enable duplicate-remover
Google Cloud Run
bashDownloadCopy code# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/your-project/duplicate-remover

# Deploy to Cloud Run
gcloud run deploy duplicate-remover \
  --image gcr.io/your-project/duplicate-remover \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2
Kubernetes Deployment
deployment.yaml:
yamlDownloadCopy codeapiVersion: apps/v1
kind: Deployment
metadata:
  name: duplicate-remover
spec:
  replicas: 3
  selector:
    matchLabels:
      app: duplicate-remover
  template:
    metadata:
      labels:
        app: duplicate-remover
    spec:
      containers:
      - name: flask-app
        image: your-registry/duplicate-remover:latest
        ports:
        - containerPort: 5000
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
---
apiVersion: v1
kind: Service
metadata:
  name: duplicate-remover-service
spec:
  selector:
    app: duplicate-remover
  ports:
  - port: 80
    targetPort: 5000
  type: LoadBalancer

⚡ Performance Optimization
For Large Datasets (>100K rows)
pythonDownloadCopy code# Use chunking for processing
def process_large_dataset(df, chunk_size=1000):
    chunks = [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
    results = []
    
    for chunk in chunks:
        result = process_chunk(chunk)
        results.append(result)
    
    return combine_results(results)

# Enable parallel processing
from joblib import Parallel, delayed

def parallel_duplicate_detection(df, n_jobs=-1):
    return Parallel(n_jobs=n_jobs)(
        delayed(detect_duplicates)(chunk) 
        for chunk in chunks
    )
Memory Optimization
pythonDownloadCopy code# Reduce memory usage
df = df.astype({
    'Age': 'int8',
    'Salary': 'int32',
    'Score': 'float32'
})

# Use sparse matrices for high-dimensional data
from scipy.sparse import csr_matrix
sparse_features = csr_matrix(features)

# Delete unnecessary data
del large_variable
import gc
gc.collect()
Caching Results
pythonDownloadCopy codefrom functools import lru_cache
import hashlib

@lru_cache(maxsize=100)
def cached_analysis(data_hash, params_hash):
    # Analysis logic here
    return results

# Usage
data_hash = hashlib.md5(df.to_csv().encode()).hexdigest()
params_hash = hashlib.md5(str(params).encode()).hexdigest()
results = cached_analysis(data_hash, params_hash)
Database Optimization
pythonDownloadCopy code# For storing results
from sqlalchemy import create_engine

engine = create_engine('sqlite:///duplicates.db')

# Batch insert results
df.to_sql('duplicates', engine, if_exists='append', 
          chunksize=1000, index=False)

# Query with indexes
engine.execute('CREATE INDEX idx_similarity ON duplicates(similarity)')
Profiling
pythonDownloadCopy codeimport cProfile
import pstats

# Profile your code
profiler = cProfile.Profile()
profiler.enable()

# Your code here
analyze_duplicates(df)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 time-consuming functions

**🐛 Troubleshooting**
Common Issues
1. "No data loaded" Error
Cause: Dataset not uploaded or session expired
Solution:
pythonDownloadCopy code# Check if data exists
if st.session_state.df is None:
    st.warning("Please upload data first")
    
# Re-upload your CSV file
# Or click "Load Sample Data"
2. Memory Error with Large Datasets
Cause: Insufficient RAM for dataset size
Solution:
pythonDownloadCopy code# Reduce dataset size
df = df.sample(n=10000)  # Random sample

# Or process in chunks
chunk_size = 1000
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    process_chunk(chunk)

# Reduce PCA components
pca = PCA(n_components=5)  # Instead of 10
3. Low Accuracy in Classification
Cause: Insufficient training examples or poor features
Solution:
pythonDownloadCopy code# Ensure enough duplicate examples
min_duplicates = 50  # At least 50 duplicate pairs

# Add more features
features.extend([
    df['col1'] * df['col2'],  # Interaction terms
    df['col1'].rolling(3).mean(),  # Rolling averages
    df.groupby('category')['value'].transform('mean')  # Group stats
])

# Try different models
for model in ['random_forest', 'svm', 'logistic_regression']:
    accuracy = detector.train_model(X, y, method=model)
    print(f"{model}: {accuracy}")
4. Slow Performance
Cause: Inefficient parameters or large dataset
Solution:
pythonDownloadCopy code# Reduce number of clusters
n_clusters = min(5, len(df) // 100)

# Use faster algorithm
method = 'kmeans'  # Instead of 'classification'

# Sample data for initial analysis
sample_df = df.sample(frac=0.1)
analysis = analyze(sample_df)

# Enable parallel processing
from sklearn.utils import parallel_backend
with parallel_backend('threading', n_jobs=-1):
    model.fit(X, y)
5. ngrok Connection Failed
Cause: Invalid auth token or network issues
Solution:
bashDownloadCopy code# Verify token
ngrok config check

# Add token manually
ngrok config add-authtoken YOUR_TOKEN

# Check connection
ngrok diagnose

# Use alternative port
ngrok http 8501 --region us
6. Streamlit Port Already in Use
Cause: Another process using port 8501
Solution:
bashDownloadCopy code# Find process using port
lsof -i :8501  # macOS/Linux
netstat -ano | findstr :8501  # Windows

# Kill process
kill -9 <PID>  # macOS/Linux
taskkill /PID <PID> /F  # Windows

# Or use different port
streamlit run streamlit_app.py --server.port 8502
7. Visualization Not Showing
Cause: Too few numeric columns or incompatible data
Solution:
pythonDownloadCopy code# Check numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
print(f"Found {len(numeric_cols)} numeric columns")

if len(numeric_cols) < 2:
    st.error("Need at least 2 numeric columns for visualization")
    
# Convert columns to numeric
df['col1'] = pd.to_numeric(df['col1'], errors='coerce')
df['col2'] = pd.to_numeric(df['col2'], errors='coerce')
8. CSV Upload Failed
Cause: File encoding or format issues
Solution:
pythonDownloadCopy code# Try different encodings
for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
    try:
        df = pd.read_csv(file, encoding=encoding)
        break
    except:
        continue

# Handle different delimiters
df = pd.read_csv(file, sep=None, engine='python')  # Auto-detect

# Skip bad lines
df = pd.read_csv(file, on_bad_lines='skip')
Debug Mode
Enable detailed logging:
pythonDownloadCopy codeimport logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.debug("Starting duplicate detection...")
Getting Help
If issues persist:

1. Check the Issues page
2. Search existing issues for similar problems
3. Create a new issue with:

Python version
Operating system
Error message
Steps to reproduce
Sample data (if possible)


**🤝 Contributing**
We welcome contributions from the community! Here's how you can help:
Ways to Contribute

* 🐛 Report Bugs: Open an issue describing the problem
* 💡 Suggest Features: Share your ideas for improvements
* 📝 Improve Documentation: Fix typos or add examples
* 🔧 Submit Code: Fix bugs or implement new features
* 🧪 Write Tests: Improve code coverage
* 🌍 Translate: Help translate the interface

**Development Setup
**bashDownloadCopy code# Fork and clone the repository
git clone https://github.com/yourusername/duplicate-remover-tool.git
cd duplicate-remover-tool

# Create development branch
git checkout -b feature/your-feature-name

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Check code style
flake8 .
black --check .

# Run type checker
mypy .
Code Style
Follow PEP 8 guidelines:
pythonDownloadCopy code# Good
def find_duplicates(data: pd.DataFrame, threshold: float = 0.9) -> List[List[int]]:
    """
    Find duplicate rows in dataframe.
    
    Args:
        data: Input dataframe
        threshold: Similarity threshold (0-1)
        
    Returns:
        List of duplicate index groups
    """
    pass

# Format with Black
black duplicate_remover_core.py

# Sort imports
isort duplicate_remover_core.py
Testing
Write tests for new features:
pythonDownloadCopy code# tests/test_duplicates.py
import pytest
from duplicate_remover_core import TextDuplicateRemover

def test_exact_duplicates():
    texts = ["hello", "hello", "world"]
    remover = TextDuplicateRemover()
    groups = remover.find_duplicates(texts)
    assert len(groups) == 1
    assert groups[0] == [0, 1]

def test_near_duplicates():
    texts = ["hello world", "hello worlds", "goodbye"]
    remover = TextDuplicateRemover(similarity_threshold=0.8)
    groups = remover.find_duplicates(texts)
    assert len(groups) >= 1
Pull Request Process

1. Create Issue: Describe what you're working on
2. Fork Repository: Create your own copy
3. Create Branch: Use descriptive branch name
4. Make Changes: Implement your feature/fix
5. Write Tests: Ensure code coverage
6. Update Docs: Document new features
7. Run Tests: Ensure all tests pass
8. Submit PR: Include description and link to issue
9. Code Review: Address feedback
10. Merge: Maintainer will merge when approved

Commit Message Format
type(scope): brief description

Detailed explanation of changes...

Fixes #123

Types: feat, fix, docs, style, refactor, test, chore
Example:
feat(classifier): add XGBoost model support

- Implemented XGBoost classifier
- Added hyperparameter tuning
- Updated model comparison function

Fixes #45


**📄 License
**This project is licensed under the MIT License. See LICENSE file for details.
MIT License

Copyright (c) 2025 [Data Fardeen]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


**💬 Support**
Documentation

* Full Documentation: docs/README.md
* API Reference: docs/API.md
* Tutorials: docs/tutorials/
* FAQ: docs/FAQ.md

**Community**

* GitHub Issues: Report bugs and request features
* Discussions: Ask questions and share ideas
* Stack Overflow: Tag questions with duplicate-remover-tool

**Commercial Support**
For enterprise support, custom development, or consulting:

* Email: fardeensidhu07@gmail.com
* Website: https://datafardeen.netlify.app
* LinkedIn: https://www.linkedin.com/in/data-fardeen-234619289/

**🙏 Acknowledgments**
This project was built using excellent open-source libraries:

* scikit-learn: Machine learning algorithms
* pandas: Data manipulation and analysis
* NumPy: Numerical computing
* Flask: Web application framework
* Streamlit: Interactive data apps
* matplotlib & seaborn: Data visualization
* Bootstrap: UI components

Special thanks to all contributors and the open-source community!


**🔒 Security
**Reporting Vulnerabilities
If you discover a security vulnerability, please email security@yourcompany.com instead of using the issue tracker. We take security seriously and will respond promptly.
Security Best Practices

* Keep dependencies updated: pip install --upgrade -r requirements.txt
* Use environment variables for sensitive data
* Enable HTTPS in production
* Implement rate limiting for public APIs
* Validate and sanitize all user inputs
* Use secure session management

Made with ❤️ for the Data Science Community
⬆ Back to Top

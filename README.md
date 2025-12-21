# Data Duplicate Remover Tool

**# Features**

The Duplicate Remover Tool provides an extensive set of advanced functionalities:

**Duplicate Detection Methods**
Exact duplicate detection
Text similarity detection using clustering
Near-duplicate numeric detection using
K-means clustering
K-Nearest Neighbors
Machine learning classification models
Logistic Regression
SVM
Decision Tree
Random Forest
Automatic similarity scoring and threshold filtering

**Visualization Features**
Cluster visualization using PCA
Model comparison charts
Distribution histograms
Box plot visualization
High quality matplotlib based charts

**Dataset Cleaning**
Automatically remove detected duplicates
Choose detection method
Download cleaned dataset as CSV
Remove exact duplicates
Remove text duplicates
Remove numeric near-duplicates
ML driven duplicate cleaning

**Two Complete Applications**
Streamlit UI powered dashboard
Flask application with modern UI
Both fully interactive and ready for deployment
Support for ngrok public URLs for cloud execution

---

**# Project Structure**
The repository contains the following core components

**duplicate_remover_core.py**
This is the heart of the system.
It includes
TextDuplicateRemover
TabularDuplicateRemover
ClassificationDuplicateDetector
CombinedDuplicateRemover
Feature extraction
Clustering
PCA dimensionality reduction
Model training
Duplicate scoring
Visualization utilities
Base64 image generator for web embedding

**streamlit_app.py**
A fully responsive Streamlit based graphical interface that offers
Dataset upload
Analysis dashboard
Cleaning interface
Visualization section
Interactive configuration controls
CSV download support

**flask_app.py**
A production style Flask application with custom HTML, CSS and Bootstrap UI
Upload datasets
Analyze duplicates
Clean datasets
Visualize results
Download cleaned CSV
AJAX powered interactive frontend
Base64 encoded charts

**Ngrok Integration**
Both Streamlit and Flask apps support ngrok public URLs
Useful for sharing your app from Google Colab or local machine

---

**# Installation**

Install all dependencies using the following command inside your environment

pip install streamlit flask flask-ngrok pyngrok pandas numpy matplotlib seaborn scikit-learn

These libraries enable
Machine learning
Data processing
Web server functionality
Interactive dashboard
Tunneling using ngrok

---

**# How to Run the Streamlit App**

Run the following command

streamlit run streamlit_app.py

Your Streamlit Duplicate Remover dashboard becomes available locally at
http localhost 8501
or at an ngrok URL when used in Google Colab

---

**# How to Run the Flask App**

Run the Flask backend using

python flask_app.py

Your Flask UI will be available at

http localhost 5000
or a public ngrok URL

---

**# Running With ngrok (Optional)**

You must first set your ngrok auth token which is available at
https ngrok.com

Example

ngrok.set_auth_token("YOUR_TOKEN_HERE")

Then you can create a secure public URL for your Streamlit or Flask server
This is especially useful when running inside Google Colab

---

**# How Duplicate Detection Works**

**Text Duplicate Detection**
Character and word based handcrafted features
PCA dimensionality reduction
K-means clustering
Similarity scoring using euclidean distance
Automatic grouping of similar text rows

**Numeric Duplicate Detection**
Standardization
PCA
Clustering based grouping
KNN based similarity discovery
Threshold based scoring

**Machine Learning Duplicate Detection**
Pair generation of feature rows
Label creation for exact matches
Training classification models to predict if two rows are duplicates
Probabilistic scoring
Grouping of high confidence duplicate pairs

---

**# Visualizations**

The tool provides multiple visual analytics including
Cluster scatter plots
Model comparison bar charts
Histogram and box plot for selected columns
Color coded clusters
Automatically generated PCA projections
Base64 encoded images for integration inside web interfaces

---

**# Cleaning Options**

Choose any method to clean
K-means
KNN
Classification
Or combine with exact duplicate removal and text based detection
Export cleaned dataset as CSV

---

**# Sample Data**

The project includes built-in sample data for quick testing
Name
Age
City
Salary
Department

---

**# Technologies Used**
Python
Streamlit
Flask
Bootstrap
NumPy
Pandas
Matplotlib
Scikit-learn
Seaborn
PCA
K-means
KNN
Random Forest
SVM
Logistic Regression
Ngrok

---

**# Use Cases**

This tool can be used for

General dataset cleaning
Removing noisy duplicates from ML data
Text deduplication for NLP
Preparing structured datasets
Analyzing numeric similarity
Identifying anomalies in tabular data
Teaching clustering and classification techniques
Research and academic projects
Data preprocessing automation

---

**# Advantages**

Supports both text and numeric data
Multiple ML methods for comparison
Interactive dashboards
Works offline or online via ngrok
Beginner friendly UI
Production ready Flask API
Advanced visual reporting
Automatic CSV exports
No complex setup required

---

**# Future Enhancements (Optional Vision)**

Deep learning based text similarity for larger datasets
Support for Excel, JSON input formats
Automated reporting in PDF format
Column level duplicate explanations
Cloud deployment template (Docker, Render, AWS)

---

**# License**

You may include your license details here such as MIT License or Apache License depending on your preference.

---

**# Author**

This project was created to provide a complete end-to-end solution for duplicate detection and dataset cleaning using machine learning, clustering and interactive visualization.

---

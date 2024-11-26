# Project3G1
# Fake News Classification
## Members
Terry Brown
Elijah Mercier
LaQuita Palmer
Ivanna Price

### Resources
WELFake_Dataset.csv:Fake News Classification Dataset by saurabhshahane
evaluation.csv, test (1).csv, train (2).csv:Fake News Classification Dataset by aadyasingh55: 

### Models Used 
LSTM-RNN
LDA
Transformers

### Questions
1. Headline vs. Content Matching/Source credibility
2. Which keywords or phrases are most commonly found in fake news?
3. What topics are targeted by fake news most?
4. Frequency and Time posted

Overview
Fake news is a pervasive issue in today's digital world, influencing public opinion and decision-making. This project aims to develop a machine learning pipeline that accurately classifies news articles as real or fake while analyzing patterns, trends, and keywords commonly found in fake news. We used a combination of advanced machine learning techniques, including LSTM-RNN, LDA, VADER Sentiment Analysis, and Transformers.

Models Used
LSTM-RNN: Built to capture sequential dependencies in text, improving our understanding of linguistic patterns.
Transformers (BERT): Fine-tuned pre-trained models to leverage contextual understanding capabilities for classification tasks.
LDA (Latent Dirichlet Allocation): Utilized for topic modeling to uncover common themes in fake news.
VADER Sentiment Analysis: Provided insights into the emotional tone of news headlines and articles.

Key Questions Explored
Headline vs. Content Matching: How aligned are headlines with the actual content?
Source Credibility: What patterns are found in fake news sources?
Common Keywords: Which keywords or phrases are most frequently found in fake news?
Targeted Topics: What topics are most often targeted by fake news?
Frequency & Timing: What are the posting frequencies and timings of fake news?

Project Goals
Develop a machine learning model to classify news articles as either "real" or "fake."
Identify patterns and trends in fake news, such as keywords, emotional sentiment, and topics.
Build a practical tool that can be used by non-technical users to verify news credibility.
Data Collection and Preparation

Datasets Used:
Collected datasets from Kaggle, including WELFake, train.csv, test.csv, and evaluation.csv.

Data Cleaning:
Removed duplicates, null values, punctuation, and stop words.
Standardized data label formats and combined datasets.

Feature Extraction:
Applied TF-IDF for vectorization and used embedding techniques for deep learning models.
Sentiment Analysis:
Used VADER to analyze the sentiment of news articles, revealing that fake news articles often had a more polarized sentiment compared to real news.
Exploratory Data Analysis (EDA)
Headline vs. Content Analysis: Identified discrepancies to detect misleading headlines.
Frequency Analysis: Identified common keywords in fake news using word clouds.
Topic Modeling: Used LDA to identify topics commonly targeted by fake news.
Posting Trends: Visualized frequency and timing patterns.
Data Splitting and Baseline Model
Split the dataset into training (80%) and testing (20%) to ensure a proper evaluation framework.
Baseline Model: Developed a logistic regression model using TF-IDF features to set a benchmark, which achieved an accuracy of approximately 75%.
Advanced Model Implementations
LSTM-RNN: Optimized with batch size, learning rate, and dropout, allowing the model to learn sequential dependencies.
Multiple LSTM and Bidirectional LSTM: Enhanced complexity to capture more nuanced temporal relationships.
Transformers (BERT): Fine-tuned for better accuracy, recall, and precision, significantly outperforming traditional ML methods.
VADER Sentiment Analysis: Incorporated sentiment scores to provide insights into emotional content, enhancing feature representations for the models.

Model Evaluation and Optimization
Evaluated models using accuracy, precision, recall, F1-score, and confusion matrix.
Iteratively fine-tuned models by adjusting:
Hyperparameters (e.g., learning rate, dropout rate).
Class balancing techniques like oversampling.
Embeddings like Word2Vec to improve feature richness.

Results:
The Transformer model achieved the highest accuracy and recall, effectively distinguishing between real and fake news articles.
The LSTM model captured temporal dependencies, making it highly effective in detecting fake news patterns.
Gradio Interface Implementation
Implemented a Gradio web interface to make the model accessible to end users without programming knowledge.
Features:
Users can input news articles or headlines to get real-time predictions.
Allows selection between different models (LSTM, Transformer) to compare results.
Provides a classification label (Real or Fake) and a confidence score.
Why Gradio?: It bridges the gap between machine learning experts and end users, providing an easy-to-use platform for journalists, educators, and the general public to verify news credibility.
Results and Conclusions
Significant Improvements: Our models showed considerable performance improvements over the baseline model. The Transformer and LSTM models achieved high accuracy, with the Transformer model being particularly robust in distinguishing between real and fake articles.
Sentiment Insights: Sentiment analysis with VADER showed that polarized articles were more likely to be fake.

Impact: The project successfully built a reliable tool for detecting fake news, which has practical applications for online media platforms and journalists to curb misinformation.

Challenges and Solutions
Initial Model Failures: No output was observed initially due to model architecture issues. Addressed by removing dropout layers, using dummy data forward passes, and adding try-except blocks.
Excessive Runtime: Optimized runtime by adjusting LSTM layer parameters and switching runtime to T4.

Next Steps
Enhance Feature Engineering: Use more advanced embeddings for richer representations.
Additional Transformer Models: Explore additional variants to improve robustness.
Real-Time System: Integrate the model into a real-time detection system for broader application.

Contributors
Terry Brown, Elijah Mercier, LaQuita Palmer, Ivanna Price
Repository Summary
This repository provides a comprehensive pipeline to classify fake news articles using advanced machine learning techniques. The README provides an overview of the models used, key project features, and future plans for improvement. Contributions are welcome for extending the project further into real-time applications and exploring new models.

Languages Used:

GOOGLE COLAB 100%


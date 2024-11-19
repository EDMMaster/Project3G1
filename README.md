# Project3G1
# Fake News Classification
## Members
Terry Brown
Elijah Mercier
LaQuita Palmer
Ivanna Price

### Models Used 
LSTM-RNN
Transformers

### Questions
1. Headline vs. Content Matching/Source credibility
2. Which keywords or phrases are most commonly found in fake news?
3. What topics are targeted by fake news most?
4. Frequency and Time posted

### Project Overview 
Fake news is a pervasive issue in today's digital world, influencing public opinion and decision-making. Our goal is to develop a machine learning pipeline that accurately classifies news articles as real or fake while analyzing patterns, trends, and keywords commonly found in fake news.

We utilized advanced machine learning methodologies, including LSTM-RNN, LDA, and Transformers, to solve this problem. The solution leverages state-of-the-art tools and libraries, along with collaborative efforts to build a reliable and scalable model.

What We Did Data Preparation Dataset Collection:

Used datasets like WELFake_Dataset.csv, train.csv, test.csv, and evaluation.csv. These datasets contained news headlines, content, and labels (real or fake). Collaborated via Google Colab for unified analysis and processing.

Data Cleaning: Removed duplicates, null values, punctuation, and stop words from the text. Filtered out irrelevant data and standardized text formats. Applied tokenization and vectorization (TF-IDF) for feature extraction. Exploratory Data Analysis (EDA):

Analyzed headline vs. content mismatches. Identified common keywords and phrases found in fake news using frequency analysis and word clouds. Investigated topics targeted by fake news using topic modeling with LDA. Visualized posting patterns (frequency and time trends) to understand distribution over time. Data Splitting:

Split datasets into training, validation, and testing sets for robust model evaluation. Model Implementation Baseline Model:

Developed a logistic regression model using TF-IDF vectors as features to set a baseline for performance comparison. Advanced Models:

LSTM-RNN: Built an LSTM-based recurrent neural network to capture sequential dependencies in the text. Optimized with hyperparameters like batch size, learning rate, and dropout for better generalization. LDA: Used Latent Dirichlet Allocation for topic modeling to uncover themes in fake and real news. Transformers: Fine-tuned a pre-trained BERT (Bidirectional Encoder Representations from Transformers) model for text classification. Compared transformer-based models with traditional ML approaches. Model Evaluation:

Evaluated models using metrics like accuracy, precision, recall, F1-score, and confusion matrix. Visualized performance metrics to compare models effectively. Optimization and Iteration Fine-tuned models iteratively by:

Adjusting hyperparameters (e.g., learning rate, dropout rate). Balancing class distribution with oversampling and under-sampling techniques. Adding embeddings like Word2Vec for richer text representations. Documented changes in performance using tables and visualizations.
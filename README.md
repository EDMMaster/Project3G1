# Project3G1
# Fake News Predictor and Classification
## Members
Terry Brown
Elijah Mercier
LaQuita Palmer
Ivanna Price

## Models Used 
LDA
VADER Sentiment Analysis
LSTM
Multiple LSTM
Bidirectional LSTM

## Questions
1. Which type of LSTM model is better at predicting whether an article is real or fake?
2. Which keywords or phrases are most commonly found in fake news?
3. What topics are targeted by fake news most?

## Project Overview 
Fake news is a pervasive issue in today's digital world, influencing public opinion and decision-making. Our goal is to develop a machine learning pipeline that accurately classifies news articles as real or fake while analyzing patterns, trends, and keywords commonly found in fake news.

### Objective:
 Develop a machine learning model to predict
 newly given news articles as either fake or real.
 Key Goals:
 Improve detection of fake news using NLP and
 LSTM techniques.
 Understand keywords and topics commonly
 associated with fake news.
 Visualize trends such as frequency, timing, and
 targeted topics.
 Applications:
 Curb misinformation in online platforms.
 Build trust in online journalism

We utilized advanced machine learning methodologies, including LSTM-RNN, LDA, and Transformers, to solve this problem. The solution leverages state-of-the-art tools and libraries, along with collaborative efforts to build a reliable and scalable model.

## PROJECT FEATURES
 Data Cleaning and Conversion: 
    Function to convert and combine topic labels into a singular DataFrame, enhancing interpretability of the results.
 Latent Dirichlet Allocation (LDA) Implementation:
    LDA Model: Initialized with 10 topics.
    Fit Model: Applied to the document-term matrix (DTM)
 Topic Analysis:
    Top Words: Extracted top 15 words for each topic to understand key themes.
    Transform DTM: DTM transformed into topic distributions for each document.
 Vader Sentiment Analysis:
    Used to analyze the sentiment of news headlines and articles, providing insights into the emotional
     tone of the content
 Topic Labels: 
    Assigned labels (e.g., Entertainment, Sports) to the most probable topic for each document.
 LSTM Model Framework:
    Usage of LSTM, Multiple LSTM, and Bidirectional LSTM models.
 Gradio Application Interface:
    Added for user-friendly design and model usage functionality.

### Data Preparation:
Used datasets like WELFake_Dataset.csv, train.csv, test.csv, and evaluation.csv. These datasets contained news headlines, content, and labels (real or fake). Collaborated via Google Colab for unified analysis and processing.

### Data Cleaning: 
Removed duplicates, null values, punctuation, and stop words from the text. Filtered out irrelevant data and standardized text formats. Applied tokenization for feature extraction.

Analyzed headline vs. content mismatches. Identified common keywords and phrases found in fake news using frequency analysis and word clouds. Investigated topics targeted by fake news using topic modeling with LDA. Visualized real and fake distributions in data.

Split datasets into training, validation, and testing sets for robust model evaluation. Model Implementation Baseline Model:

### LSTM: 
Built multiple LSTM models for model differential analysis. Optimized with hyperparameters like batch size, learning rate, and dropout for better generalization. LDA: Used Latent Dirichlet Allocation for topic modeling to uncover themes in fake and real news. 

Evaluated models using metrics such as accuracy and confusion matrix. Visualized performance metrics to compare models effectively. 

Adjusting hyperparameters (e.g., learning rate, dropout rate, epoch count). Balancing class distribution with oversampling and under-sampling techniques. Documented changes in performance using tables and visualizations.

### Gradio Interface:
Was used in order to interact with models and for real-case usage. LSTM, Multiple LSTM, and Bidirectional LSTM models are able to be selected before entering a news article to be predicted with. The model will then output its prediction followed by its condifence in the prediction.

## Results and Conclusions
### Results:
Our results demonstrated significant improvements over the baseline model, particularly when utilizing
advanced deep learning models.

All LSTM models reached high accuracies, but it struggled with nuanced
linguistic patterns that weren't included in the dataset often found in current fake news.
The LSTM and Multiple LSTM models achieved significantly better performance compared to the
logistic regression model, with the Multiple LSTM model achieving the highest accuracy and recall.

The LSTM model was particularly confident at capturing results, which helped improve detection of fake
news patterns over simpler models.

Common keywords and topics targeted by fake news were identified, which helped reveal the strategies
and narratives commonly employed by fake news publishers. These insights can be used to inform future mitigation strategies and content monitoring.

### Conclusion
The Multiple LSTM model provided superior accuracy and robustness for fake news detection.
Its contextual understanding capabilities made it ideal for distinguishing between real and fake news, even
when similar language was used.

## Challenges:
 1. Initial models had no output. This was resolved by removing the dropout
 layers, adding try and except blocks, and using a dummy data forward pass. 
 Removing the dropout layer contributed to resolving the issue by allowing
 the model to utilize its full learning capacity and providing a stable,
 uninterrupted learning process.
 The try-except blocks offer clear error messages, and the dummy data
 forward pass ensures the model layers are correctly built.
 These modifications prevented scenarios where the models failed silently,
 leading to no output. 
 1. Excessive run time. Solved by adjusting parameters such as # of units, #
 of epochs, batch size, etc. in each LSTM layer. Also changed runtime type
 to T4.
 1. model_1.keras
## Future Steps to Expand on Project:
 Analyze VADER results to explain emotional
 sentiment findings visually.
 F1-Score, Recall, and Precision parameters and
 analysis.
 Further epoch training.
 Visualize trends such as frequency, timing, and
 targeted topics.
 Explore additional transformer models.
 Enhance features using more advanced embeddings.
 Integrate the model into a real-time detection system.
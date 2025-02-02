# SMS Spam Detection Using Machine Learning

## Overview

This project implements a machine learning solution for detecting SMS spam messages. It uses text preprocessing techniques, TF-IDF vectorization, and a **Multinomial Naive Bayes (MNB)** classifier to distinguish between spam and non-spam (ham) messages. The project is deployed as a web application using Flask with a front-end built using HTML and CSS.

## Need for This Project

With the exponential growth of digital communication, SMS and messaging platforms have become essential in daily life. However, the increase in unsolicited messages (spam) poses several challenges:
- **User Experience:** Spam messages clutter inboxes, causing inconvenience and potentially exposing users to phishing and fraudulent schemes.
- **Security Concerns:** Many spam messages are designed to trick users into providing sensitive information, increasing the risk of cyberattacks.
- **Resource Waste:** Spam can lead to unnecessary consumption of network bandwidth and storage.
  
Automating spam detection is crucial to protect users and enhance their messaging experience. Traditional rule-based methods struggle to keep pace with evolving spam tactics. Machine learning, especially using text-based models, offers a dynamic and scalable approach to accurately detect and filter out spam messages.

## Facts About SMS Spam

- **Volume:** Millions of spam messages are sent daily worldwide, affecting both individual users and businesses.
- **Evolving Tactics:** Spammers continuously update their strategies to bypass traditional filters.
- **Cost:** Spam not only affects user experience but also incurs significant costs related to data usage and system maintenance.
- **Security Risk:** A substantial number of spam messages aim to steal personal and financial information through phishing and malware distribution.

## Role of Machine Learning in Span Detection

Machine Learning plays a pivotal role in automating the detection of spam messages. Its benefits include:
- **Adaptive Learning:** ML models can learn from data, automatically adapting to new patterns and spam tactics.
- **Scalability:** ML-based approaches can handle large volumes of data efficiently.
- **High Accuracy:** By leveraging statistical and linguistic features of text, ML algorithms can accurately distinguish between spam and non-spam.
- **Continuous Improvement:** As more data becomes available, the model can be retrained to further improve its performance.
- **Real-time Predictions:** Once deployed, the ML model can provide instant predictions, enhancing user experience by filtering spam in real time.

In this project, the **Multinomial Naive Bayes (MNB) classifier** is chosen due to its efficiency and performance in text classification tasks. The project also tested several algorithms (as shown in the performance metrics) to understand their accuracy and precision for SMS spam detection.

## How It Works

### Data Preprocessing & Model Training
- The script loads the SMS spam dataset.
- It preprocesses the text by cleaning and stemming the messages using NLTK.
- The preprocessed text is then vectorized using TF-IDF.
- A **Multinomial Naive Bayes classifier** is trained on this vectorized data.
- The trained model and the TF-IDF vectorizer are saved using Python's pickle module.

### Prediction API
- The Flask API listens for POST requests on the `/predict` endpoint.
- It accepts a JSON payload containing a `message` field.
- The input text is preprocessed (cleaned and stemmed) and then vectorized using the saved TF-IDF vectorizer.
- The saved **Multinomial Naive Bayes model** is used to predict whether the message is spam or not.
- The result ("Spam" or "Not Spam") is returned as a JSON response.

### Web Interface
- A simple web interface built with HTML and CSS is provided.
- Users can enter SMS messages into an HTML form.
- The form sends the message to the Flask API.
- The prediction result is displayed instantly on the web interface.

## Performance Metrics

The following table summarizes the accuracy and precision of various machine learning algorithms tested in the project:

| Algorithm | Accuracy   | Precision  |
|-----------|------------|------------|
| RF (Random Forest)        | 0.972921   | 0.982456   |
| ETC (Extra Trees Classifier)       | 0.977756   | 0.975207   |
| XGB (XGBoost)       | 0.974855   | 0.951613   |
| GBDT (Gradient Boosting Decision Trees)      | 0.949710   | 0.938776   |
| BgC (Bagging Classifier)       | 0.970019   | 0.934959   |
| AdaBoost  | 0.965184   | 0.925000   |
| LR (Logistic Regression)        | 0.949710   | 0.877193   |
| DT (Decision Tree)        | 0.941973   | 0.809524   |
| KN (K-Nearest Neighbors)        | 0.920696   | 0.741379   |
| SVC (Support Vector Classifier)       | 0.866538   | 0.000000   |

### **Best Performing Model**
Based on accuracy and precision, the **Random Forest (RF) model** demonstrated the best performance, achieving **97.29% accuracy** and **98.24% precision**. While the **Extra Trees Classifier (ETC)** and **XGBoost (XGB)** also performed exceptionally well, RF was the most balanced model in terms of overall metrics.  

However, for this project, **Multinomial Naive Bayes (MNB) was chosen** due to its efficiency and effectiveness in text classification tasks, making it an optimal choice for SMS spam detection.

## Deployment

The project is deployed using Flask for the backend API and a simple front-end built with HTML and CSS. The web application allows users to submit SMS messages through a form and receive a prediction indicating whether the message is "Spam" or "Not Spam".

## Installation and Setup

### Prerequisites

- Python 3.7 or higher
- Git
- Refer requirements.txt file

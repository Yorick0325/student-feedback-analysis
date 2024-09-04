import os
import re
import warnings
import docx
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from textblob import TextBlob

warnings.filterwarnings("ignore")

# Ensure NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('punkt')


# Data reading functions
def read_word_file(file_path):
    doc = docx.Document(file_path)
    data = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            data.append(text)
    return data


# Preprocessing function
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())  # Lowercase and remove punctuation
    words = [PorterStemmer().stem(word) for word in word_tokenize(text) if word not in stopwords.words('english')]
    return ' '.join(words)


# Perform sentiment analysis and auto-labeling
def perform_sentiment_analysis(text_data):
    sentiments = []
    for text in text_data:
        analysis = TextBlob(text)
        if analysis.sentiment.polarity > 0:
            sentiments.append(1)  # Positive
        elif analysis.sentiment.polarity == 0:
            sentiments.append(2)  # Neutral
        else:
            sentiments.append(0)  # Negative
    return sentiments


# Main processing function with auto-labeling
def process_feedback_for_auto_labeling(file_path):
    feedback_data = read_word_file(file_path)
    preprocessed_data = [preprocess_text(text) for text in feedback_data]
    sentiments = perform_sentiment_analysis(preprocessed_data)

    # Save results
    df = pd.DataFrame({'feedback': feedback_data, 'preprocessed_answer': preprocessed_data, 'sentiment': sentiments})
    df.to_csv('feedback_with_sentiment.csv', index=False)
    return df


# Main function for auto-labeling
if __name__ == "__main__":
    file_path = '/Users/wanwan/Desktop/data2/overall evaluation.docx'  
    df = process_feedback_for_auto_labeling(file_path)

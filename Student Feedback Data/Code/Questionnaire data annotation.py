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

# Globally ignore all warnings
warnings.filterwarnings("ignore")

# Ensure NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('punkt')

# Data reading functions
def read_word_file(file_path):
    doc = docx.Document(file_path)
    data = []
    question, answer = "", ""
    for para in doc.paragraphs:
        text = para.text.strip()
        if re.match(r'Q\d', text):
            if question and answer:
                data.append({'question': question, 'answer': answer})
            question, answer = text, ""
        else:
            answer += " " + text if answer else text
    if question and answer:
        data.append({'question': question, 'answer': answer})
    return data

def read_all_word_files(directory):
    all_data = []
    for filename in os.listdir(directory):
        if filename.endswith(".docx"):
            file_path = os.path.join(directory, filename)
            all_data.extend(read_word_file(file_path))
    return all_data

# Preprocessing function
def preprocess_text(text):
    empty_responses = ['none provided', '(empty)', 'no answer', 'no response', 'nope', 'no', 'empty_response']
    if any(phrase in text.lower() for phrase in empty_responses):
        return ''
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())  # Lowercase and remove punctuation
    words = [PorterStemmer().stem(word) for word in word_tokenize(text) if word not in stopwords.words('english')]
    return ' '.join(words)

# Main processing function with auto-labeling and manual correction
def process_feedback_for_auto_labeling(directory):
    feedback_data = read_all_word_files(directory)
    df = pd.DataFrame(feedback_data)

    if 'answer' not in df.columns:
        print("No 'answer' column found in the data.")
        return df

    df['preprocessed_answer'] = df['answer'].apply(preprocess_text)

    # Ensure 'preprocessed_answer' column is generated
    if 'preprocessed_answer' not in df.columns or df['preprocessed_answer'].isnull().all():
        print("Preprocessing failed: 'preprocessed_answer' column not found or is empty.")
        return df

    # Initialize an empty DataFrame to collect all sentiments
    questions = {
        'Q1': df[df['question'].str.contains('Q1')].copy(),
        'Q2': df[df['question'].str.contains('Q2')].copy(),
        'Q3': df[df['question'].str.contains('Q3')].copy(),
        'Q4': df[df['question'].str.contains('Q4')].copy()
    }

    for q, feedback in questions.items():
        print(f"\nProcessing {q}...")

        feedback = feedback[feedback['preprocessed_answer'] != '']

        if feedback.empty:
            print(f"No valid responses for {q} after preprocessing.")
            continue

        # Auto-labeling step
        feedback['auto_label'] = feedback['preprocessed_answer'].apply(lambda x: TextBlob(x).sentiment.polarity)
        feedback['auto_label'] = feedback['auto_label'].apply(lambda polarity: 1 if polarity > 0 else (0 if polarity < 0 else 2))

        # Export auto-labeled results to CSV for manual correction
        feedback.to_csv(f'{q}_feedback_with_auto_labels.csv', index=False)
        print(f"Auto-labeled data for {q} saved to {q}_feedback_with_auto_labels.csv. Please manually correct the labels and reload.")

    return questions

# Main function for auto-labeling
if __name__ == "__main__":
    directory = '/Users/wanwan/Desktop/data1'
    process_feedback_for_auto_labeling(directory)

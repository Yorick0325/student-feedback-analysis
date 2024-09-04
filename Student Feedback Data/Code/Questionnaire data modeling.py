import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from transformers import BertTokenizer, BertModel
from wordcloud import WordCloud
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary

# Globally ignore all warnings
warnings.filterwarnings("ignore")

# Batch BERT feature extraction with batch processing
def get_bert_embeddings_batch(text_list, batch_size=32):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    embeddings = []
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

# Visualization functions
def generate_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400).generate(' '.join(text))
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

def display_topics(model, feature_names, no_top_words):
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        topics[f"Topic {topic_idx + 1}"] = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
        print(f"Topic {topic_idx + 1}:")
        print(" ".join(topics[f"Topic {topic_idx + 1}"]))
    return topics

# Function: Test perplexity and coherence for different numbers of topics
def evaluate_topic_modeling(text_data, sentiments, min_topics=2, max_topics=10, no_top_words=10, max_df=0.95, min_df=2, alpha=0.1, eta=0.01):
    perplexities = []
    coherences = []
    topic_range = range(min_topics, max_topics + 1)

    for num_topics in topic_range:
        print(f"Evaluating for {num_topics} topics...")
        vectorizer = CountVectorizer(max_df=max_df, min_df=min_df, stop_words='english')
        X = vectorizer.fit_transform(text_data)

        lda = LatentDirichletAllocation(n_components=num_topics, max_iter=50, learning_method='online',
                                        random_state=0, doc_topic_prior=alpha, topic_word_prior=eta)
        lda.fit(X)

        # Calculate perplexity
        perplexity = lda.perplexity(X)
        perplexities.append(perplexity)

        # Calculate coherence
        feature_names = vectorizer.get_feature_names_out()
        topics_tokens = [[feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]] for topic in lda.components_]
        dictionary = Dictionary([text.split() for text in text_data])
        corpus = [dictionary.doc2bow(text.split()) for text in text_data]
        coherence_model_lda = CoherenceModel(topics=topics_tokens, texts=[text.split() for text in text_data],
                                             dictionary=dictionary, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        coherences.append(coherence_lda)

        print(f"Perplexity for {num_topics} topics: {perplexity}")
        print(f"Coherence for {num_topics} topics: {coherence_lda}")

    # Select the number of topics with the highest coherence score
    optimal_topics = topic_range[np.argmax(coherences)]
    print(f"Optimal number of topics based on coherence: {optimal_topics}")

    return optimal_topics, perplexities, coherences

# Function: Visualize the relationship between perplexity/coherence and the number of topics
def plot_perplexity_coherence(topic_range, perplexities, coherences):
    plt.figure(figsize=(14, 7))

    # Plot perplexity
    plt.subplot(1, 2, 1)
    plt.plot(topic_range, perplexities, marker='o', color='b')
    plt.title('Perplexity vs. Number of Topics')
    plt.xlabel('Number of Topics')
    plt.ylabel('Perplexity')

    # Plot coherence
    plt.subplot(1, 2, 2)
    plt.plot(topic_range, coherences, marker='o', color='r')
    plt.title('Coherence vs. Number of Topics')
    plt.xlabel('Number of Topics')
    plt.ylabel('Coherence Score')

    plt.tight_layout()
    plt.show()

# Function for topic modeling with sentiment distribution and validation
def perform_topic_modeling(text_data, sentiments, num_topics=5, no_top_words=10, max_df=0.95, min_df=2, alpha=0.1, eta=0.01):
    vectorizer = CountVectorizer(max_df=max_df, min_df=min_df, stop_words='english')
    X = vectorizer.fit_transform(text_data)

    lda = LatentDirichletAllocation(n_components=num_topics, max_iter=50, learning_method='online',
                                    random_state=0, doc_topic_prior=alpha, topic_word_prior=eta)
    lda.fit(X)

    # Calculate perplexity
    perplexity = lda.perplexity(X)
    print(f"Model Perplexity: {perplexity}")

    topics = display_topics(lda, vectorizer.get_feature_names_out(), no_top_words)

    # Calculate topic distribution
    topic_distribution = lda.transform(X)

    # Initialize a dictionary to store sentiment distribution for each topic
    topic_sentiment_distribution = {f"Topic {i+1}": {'Positive': 0, 'Negative': 0, 'Neutral': 0} for i in range(num_topics)}

    # Calculate sentiment tendency for each topic
    for i, dist in enumerate(topic_distribution):
        topic_idx = dist.argmax()  # Get the most probable topic index for each text
        sentiment = sentiments[i]  # Get the sentiment label for that text
        if sentiment == 1:
            topic_sentiment_distribution[f"Topic {topic_idx + 1}"]['Positive'] += 1
        elif sentiment == 0:
            topic_sentiment_distribution[f"Topic {topic_idx + 1}"]['Negative'] += 1
        elif sentiment == 2:
            topic_sentiment_distribution[f"Topic {topic_idx + 1}"]['Neutral'] += 1

    # Print sentiment tendency for each topic
    for topic, sentiment_dist in topic_sentiment_distribution.items():
        print(f"\n{topic} Sentiment Distribution:")
        for sentiment, count in sentiment_dist.items():
            print(f"{sentiment}: {count}")

    # Calculate topic coherence
    feature_names = vectorizer.get_feature_names_out()
    topics_tokens = [[feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]] for topic in lda.components_]
    dictionary = Dictionary([text.split() for text in text_data])
    corpus = [dictionary.doc2bow(text.split()) for text in text_data]
    coherence_model_lda = CoherenceModel(topics=topics_tokens, texts=[text.split() for text in text_data],
                                         dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print(f"Model Coherence: {coherence_lda}")

    return topic_distribution, topics, topic_sentiment_distribution

# Compile and train LSTM model
def compile_and_train_lstm_model(X_train, y_train, X_test, y_test):
    model = Sequential()
    model.add(Embedding(input_dim=10000, output_dim=128, input_length=100))
    model.add(LSTM(128))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    checkpoint = ModelCheckpoint(filepath='best_model.keras', monitor='val_loss', save_best_only=True)

    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1,
                        callbacks=[early_stopping, checkpoint])

    loss, accuracy = model.evaluate(X_test, y_test)
    return model, history, accuracy

# Deep Learning Model (LSTM) for individual questions
def build_and_train_lstm(feedback, q, tokenizer):
    if 'sentiment' not in feedback.columns:
        print(f"No 'sentiment' column found for {q}. Exiting.")
        return feedback

    sequences = tokenizer.texts_to_sequences(feedback['preprocessed_answer'])
    X_seq = pad_sequences(sequences, maxlen=100)

    # Apply new sentiment label logic
    y = feedback['sentiment'].apply(lambda x: 1 if x == 1 else (0 if x == 0 else 2))

    X_train, X_test, y_train, y_test = train_test_split(X_seq, y, test_size=0.2, random_state=0)

    model, history, accuracy = compile_and_train_lstm_model(X_train, y_train, X_test, y_test)
    print(f"Deep Learning Model Accuracy for {q}: {accuracy:.2f}")

    # Predict sentiment classification and store in feedback
    y_pred = np.argmax(model.predict(X_seq), axis=-1)
    feedback['predicted_sentiment'] = y_pred

    # Visualize training and validation loss/accuracy
    plt.figure(figsize=(12, 6))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Model Accuracy for {q}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Model Loss for {q}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Visualize sentiment prediction distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='predicted_sentiment', data=feedback)
    plt.title(f'Predicted Sentiment Distribution for {q}')
    plt.xlabel('Predicted Sentiment Class')
    plt.ylabel('Frequency')
    plt.show()

    return feedback

# Visualize sentiment analysis and percentage
def visualize_sentiment_distribution(df, title):
    sentiment_counts = df['sentiment'].value_counts(normalize=True) * 100
    labels = ['Positive', 'Negative', 'Neutral']

    plt.figure(figsize=(8, 8))
    plt.pie(sentiment_counts, labels=labels, autopct='%1.1f%%', startangle=140,
            colors=['#66b3ff', '#ff6666', '#ffcc99'])
    plt.title(f'Sentiment Distribution for {title}')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()

# Overall Deep Learning Model (LSTM)
def build_and_train_overall_lstm(df, tokenizer):
    if 'sentiment' not in df.columns:
        print("No 'sentiment' column found. Exiting.")
        return df

    sequences = tokenizer.texts_to_sequences(df['preprocessed_answer'])
    X_seq = pad_sequences(sequences, maxlen=100)

    y = df['sentiment'].apply(lambda x: 1 if x == 1 else (0 if x == 0 else 2))  # Positive: 1, Negative: 0, Neutral: 2

    X_train, X_test, y_train, y_test = train_test_split(X_seq, y, test_size=0.2, random_state=0)

    model, history, accuracy = compile_and_train_lstm_model(X_train, y_train, X_test, y_test)
    print(f"Overall Deep Learning Model Accuracy: {accuracy:.2f}")

    # Predict sentiment classification and store in df
    y_pred = np.argmax(model.predict(X_seq), axis=-1)
    df['predicted_sentiment'] = y_pred

    # Visualize training and validation loss/accuracy
    plt.figure(figsize=(12, 6))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Overall Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Overall Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Visualize sentiment prediction distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='predicted_sentiment', data=df)
    plt.title('Overall Predicted Sentiment Distribution')
    plt.xlabel('Predicted Sentiment Class')
    plt.ylabel('Frequency')
    plt.show()

    return df

# Main function: Run topic modeling analysis and automatically select the optimal number of topics
if __name__ == "__main__":
    # Update the file paths with GitHub raw URLs
    questions = {
        'Q1': pd.read_csv('https://raw.githubusercontent.com/Yorick0325/student-feedback-analysis/main/Student%20Feedback%20Data/Questionnaire%20data/Manually%20corrected%20labeled%20data/Q1_feedback_with_corrected_labels.csv'),
        'Q2': pd.read_csv('https://raw.githubusercontent.com/Yorick0325/student-feedback-analysis/main/Student%20Feedback%20Data/Questionnaire%20data/Manually%20corrected%20labeled%20data/Q2_feedback_with_corrected_labels.csv'),
        'Q3': pd.read_csv('https://raw.githubusercontent.com/Yorick0325/student-feedback-analysis/main/Student%20Feedback%20Data/Questionnaire%20data/Manually%20corrected%20labeled%20data/Q3_feedback_with_corrected_labels.csv'),
        'Q4': pd.read_csv('https://raw.githubusercontent.com/Yorick0325/student-feedback-analysis/main/Student%20Feedback%20Data/Questionnaire%20data/Manually%20corrected%20labeled%20data/Q4_feedback_with_corrected_labels.csv')
    }

    combined_df = pd.DataFrame()  # Initialize an empty DataFrame

    tokenizer = Tokenizer(num_words=10000)
    all_text = [text for q, feedback in questions.items() for text in feedback['preprocessed_answer']]
    tokenizer.fit_on_texts(all_text)

    for q, feedback in questions.items():
        print(f"\n{q} Data Overview:")
        print(feedback.head())  # Print the first few rows of data
        print(feedback.columns)  # Print column names

        # Check if sentiment column exists and is not empty
        if 'sentiment' not in feedback.columns or feedback['sentiment'].isnull().all():
            print(f"No valid feedback data or sentiment column missing for {q}.")
            continue
        else:
            # Visualize annotated sentiment analysis and proportions
            visualize_sentiment_distribution(feedback, q)

            # Perform topic modeling analysis and automatically select the best number of topics
            optimal_topics, perplexities, coherences = evaluate_topic_modeling(
                feedback['preprocessed_answer'], feedback['sentiment'],
                min_topics=2, max_topics=10, no_top_words=10, max_df=0.95, min_df=2, alpha=0.1, eta=0.01)

            # Visualize the relationship between perplexity/coherence and the number of topics
            plot_perplexity_coherence(range(2, 11), perplexities, coherences)

            # Re-run topic analysis with the optimal number of topics and get sentiment distribution, perplexity, and topic coherence validation
            topic_distribution, topics, topic_sentiment_distribution = perform_topic_modeling(
                feedback['preprocessed_answer'], feedback['sentiment'], num_topics=optimal_topics,
                max_df=0.95, min_df=2, alpha=0.1, eta=0.01)

            feedback['topic'] = topic_distribution.argmax(axis=1)

            # Print topics, keywords, and sentiment tendency
            print(f"\nIdentified topics and their sentiment distribution for {q}:")
            for topic, keywords in topics.items():
                print(f"{topic}: {keywords}")
                print(f"Sentiment Distribution: {topic_sentiment_distribution[topic]}")

            # Build and train LSTM model for each question
            feedback = build_and_train_lstm(feedback, q, tokenizer)
            combined_df = pd.concat([combined_df, feedback])  # Concatenate feedback from each question into the combined dataset

    # Build and train the overall LSTM model on the combined dataset
    combined_df = build_and_train_overall_lstm(combined_df, tokenizer)

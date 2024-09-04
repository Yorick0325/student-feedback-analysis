import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
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


# Automatically determine the optimal number of topics based on coherence and perplexity
def find_optimal_num_topics(text_data, min_topics=2, max_topics=10, step=1, no_top_words=10, max_df=0.95, min_df=2,
                            alpha=0.1, eta=0.01):
    vectorizer = CountVectorizer(max_df=max_df, min_df=min_df, stop_words='english')
    X = vectorizer.fit_transform(text_data)

    coherence_values = []
    perplexity_values = []
    topic_range = range(min_topics, max_topics + 1, step)

    for num_topics in topic_range:
        lda = LatentDirichletAllocation(n_components=num_topics, max_iter=50, learning_method='online',
                                        random_state=0, doc_topic_prior=alpha, topic_word_prior=eta)
        lda.fit(X)

        perplexity = lda.perplexity(X)
        perplexity_values.append(perplexity)

        feature_names = vectorizer.get_feature_names_out()
        texts = [text.split() for text in text_data]
        dictionary = Dictionary(texts)

        coherence_model_lda = CoherenceModel(
            topics=[[feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]] for topic in lda.components_],
            texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        coherence_values.append(coherence_lda)

    # Plot coherence and perplexity
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(topic_range, coherence_values, marker='o')
    plt.title('Coherence Score by Number of Topics')
    plt.xlabel('Number of Topics')
    plt.ylabel('Coherence Score')

    plt.subplot(1, 2, 2)
    plt.plot(topic_range, perplexity_values, marker='o')
    plt.title('Perplexity by Number of Topics')
    plt.xlabel('Number of Topics')
    plt.ylabel('Perplexity')

    plt.tight_layout()
    plt.show()

    # Choose the optimal number of topics based on highest coherence and lowest perplexity
    optimal_topics = topic_range[np.argmax(coherence_values)]
    print(f"Optimal number of topics: {optimal_topics}")

    return optimal_topics


# Topic modeling with sentiment distribution and validation
def perform_topic_modeling(text_data, sentiments, num_topics=5, no_top_words=10, max_df=0.95, min_df=2, alpha=0.1,
                           eta=0.01):
    vectorizer = CountVectorizer(max_df=max_df, min_df=min_df, stop_words='english')
    X = vectorizer.fit_transform(text_data)

    lda = LatentDirichletAllocation(n_components=num_topics, max_iter=50, learning_method='online',
                                    random_state=0, doc_topic_prior=alpha, topic_word_prior=eta)
    lda.fit(X)

    perplexity = lda.perplexity(X)
    print(f"Model Perplexity: {perplexity}")

    feature_names = vectorizer.get_feature_names_out()  # Get feature names
    topics = display_topics(lda, feature_names, no_top_words)
    topic_distribution = lda.transform(X)

    # Sentiment distribution by topic
    topic_sentiment_distribution = {f"Topic {i + 1}": {'Positive': 0, 'Negative': 0, 'Neutral': 0} for i in
                                    range(num_topics)}
    for i, dist in enumerate(topic_distribution):
        topic_idx = dist.argmax()
        sentiment = sentiments[i]
        if sentiment == 1:
            topic_sentiment_distribution[f"Topic {topic_idx + 1}"]['Positive'] += 1
        elif sentiment == 0:
            topic_sentiment_distribution[f"Topic {topic_idx + 1}"]['Negative'] += 1
        elif sentiment == 2:
            topic_sentiment_distribution[f"Topic {topic_idx + 1}"]['Neutral'] += 1

    for topic, sentiment_dist in topic_sentiment_distribution.items():
        print(f"\n{topic} Sentiment Distribution:")
        for sentiment, count in sentiment_dist.items():
            print(f"{sentiment}: {count}")

    # Create dictionary
    texts = [text.split() for text in text_data]
    dictionary = Dictionary(texts)

    # Calculate topic coherence
    coherence_model_lda = CoherenceModel(
        topics=[[feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]] for topic in lda.components_],
        texts=texts, dictionary=dictionary, coherence='c_v')
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

    # Plot accuracy and loss
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return model, history, accuracy


# Build and train LSTM model for individual questions
def build_and_train_lstm(df, tokenizer):
    sequences = tokenizer.texts_to_sequences(df['preprocessed_answer'])
    X_seq = pad_sequences(sequences, maxlen=100)

    y = df['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(X_seq, y, test_size=0.2, random_state=0)

    model, history, accuracy = compile_and_train_lstm_model(X_train, y_train, X_test, y_test)
    print(f"Deep Learning Model Accuracy: {accuracy:.2f}")

    return df


# Function to generate sentiment distribution pie chart
def generate_sentiment_pie_chart(df):
    sentiment_counts = df['sentiment'].value_counts()
    labels = ['Positive', 'Negative', 'Neutral']
    plt.figure(figsize=(8, 8))
    plt.pie(sentiment_counts, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title('Sentiment Distribution')
    plt.show()


# Main function
if __name__ == "__main__":
    # Update the file path to GitHub raw link
    df = pd.read_csv('https://raw.githubusercontent.com/Yorick0325/student-feedback-analysis/main/Student%20Feedback%20Data/Overall%20Evaluation%20data/Manually%20corrected%20labeled%20data.csv')

    # Display a sentiment pie chart directly from the dataset
    generate_sentiment_pie_chart(df)

    # Tokenizer for text data
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(df['preprocessed_answer'])

    # Build and train LSTM
    df = build_and_train_lstm(df, tokenizer)

    # Find optimal number of topics
    optimal_topics = find_optimal_num_topics(df['preprocessed_answer'])

    # Perform topic modeling with the optimal number of topics
    topic_distribution, topics, topic_sentiment_distribution = perform_topic_modeling(
        df['preprocessed_answer'], df['sentiment'], num_topics=optimal_topics)

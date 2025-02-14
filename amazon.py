

import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

def preprocess_text(text):
    """Preprocess the input text"""
    # Convert to lowercase and remove special characters
    text = re.sub(r'[^\w\s]', '', str(text).lower())
    
    # Tokenization
    tokens = text.split()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Stemming and Lemmatization
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    tokens = [stemmer.stem(word) for word in tokens]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

def train_model():
    """Train the model with sample data"""
    # Sample data - you can replace this with your actual training data
    sample_data = [
        {"text": "This product is excellent!", "sentiment": "Positive"},
        {"text": "Terrible experience, very disappointed", "sentiment": "Negative"},
        {"text": "It's okay, nothing special", "sentiment": "Neutral"},
        # Add more sample data as needed
    ]
    
    df = pd.DataFrame(sample_data)
    
    # Preprocess text
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['processed_text'])
    
    # Convert sentiment to numerical values
    sentiment_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    y = df['sentiment'].map(sentiment_map)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, vectorizer

def predict_sentiment(text, model, vectorizer):
    """Predict sentiment for given text"""
    # Preprocess the input text
    processed_text = preprocess_text(text)
    
    # Vectorize the processed text
    text_vectorized = vectorizer.transform([processed_text])
    
    # Predict
    prediction = model.predict(text_vectorized)[0]
    
    # Map prediction to sentiment
    sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    return sentiment_map[prediction]

# Streamlit UI
def main():
    st.title("Review Sentiment Analysis")
    
    # Create a text input box
    review_text = st.text_area("Enter your review text:", height=100)
    
    # Create a predict button
    if st.button("Predict Review"):
        if review_text.strip() == "":
            st.warning("Please enter some text.")
        else:
            try:
                # Train model (in production, you would load a pre-trained model)
                model, vectorizer = train_model()
                
                # Make prediction
                sentiment = predict_sentiment(review_text, model, vectorizer)
                
                # Display result with appropriate color
                if sentiment == 'Positive':
                    st.success(f"This review is {sentiment} 😊")
                elif sentiment == 'Negative':
                    st.error(f"This review is {sentiment} 😞")
                else:
                    st.info(f"This review is {sentiment} 😐")
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
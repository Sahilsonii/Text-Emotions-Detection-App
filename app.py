import streamlit as st
import numpy as np
import re
from nltk.stem import PorterStemmer
import pickle
import nltk

# Download NLTK stopwords
nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words('english'))

# Load saved models
lg = pickle.load(open('logistic_regresion.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
lb = pickle.load(open('label_encoder.pkl', 'rb'))

def clean_text(text):
    stemmer = PorterStemmer()
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower().split()
    text = [stemmer.stem(word) for word in text if word not in stopwords]
    return " ".join(text)

def predict_emotion(input_text):
    cleaned_text = clean_text(input_text)
    input_vectorized = tfidf_vectorizer.transform([cleaned_text])
    predicted_label = lg.predict(input_vectorized)[0]
    predicted_emotion = lb.inverse_transform([predicted_label])[0]
    return predicted_emotion

# Streamlit UI
st.set_page_config(page_title="Emotion Classifier", layout="centered")

st.markdown("""
    <style>
        /* Animated background */
        body {
            background: linear-gradient(270deg, #ff7e5f, #feb47b, #86a8e7, #91eae4);
            background-size: 800% 800%;
            animation: gradientAnimation 15s ease infinite;
        }
        @keyframes gradientAnimation {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }
        /* Title styling with live glow effect */
        .main-title {
            font-size: 3em;
            font-weight: bold;
            text-align: center;
            color: #fff;
            margin-bottom: 20px;
            animation: textGlow 1.5s infinite alternate;
        }
        @keyframes textGlow {
            from { text-shadow: 0 0 10px #fff, 0 0 20px #fff, 0 0 30px #ff4da6, 0 0 40px #ff4da6, 0 0 50px #ff4da6, 0 0 60px #ff4da6, 0 0 50px #ff4da6; }
            to { text-shadow: 0 0 20px #fff, 0 0 30px #ff4da6, 0 0 40px #ff4da6, 0 0 50px #ff4da6, 0 0 60px #ff4da6, 0 0 70px #ff4da6, 0 0 50px #ff4da6; }
        }
        /* Other UI elements */
        .sub-title {
            font-size: 1.5em;
            text-align: center;
            color: #eee;
            margin-bottom: 40px;
        }
        .emotion-box {
            padding: 30px;
            border-radius: 12px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: #fff;
            text-align: center;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
            margin-top: 30px;
        }
        .analyze-button {
            display: block;
            width: 100%;
            padding: 12px;
            background-color: #4CAF50;
            color: #fff;
            border: none;
            border-radius: 8px;
            font-size: 1.2em;
            cursor: pointer;
            transition: background 0.3s;
        }
        .analyze-button:hover {
            background-color: #45a049;
        }
        .text-area {
            width: 100%;
            padding: 12px;
            border: 2px solid #4CAF50;
            border-radius: 8px;
            font-size: 1.1em;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Emotion Detection App</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Classify text into one of six human emotions</div>', unsafe_allow_html=True)

st.markdown("**Emotions:** Joy, Fear, Anger, Love, Sadness, Surprise")

user_input = st.text_area("Enter text to analyze:", height=150, key='input', help="Type something to detect its emotion")

if st.button("Analyze Emotion", key='analyze', help="Click to predict the emotion of the entered text"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        predicted_emotion = predict_emotion(user_input)
        st.markdown(f'<div class="emotion-box"><h3>Predicted Emotion: {predicted_emotion}</h3></div>', unsafe_allow_html=True)

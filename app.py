import streamlit as st
import joblib
import re

# Load saved components
model = joblib.load("sentiment_svm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Text cleaning function (same as used during training)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)  # remove HTML
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # keep only letters
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# App title
st.title("🛍️ Flipkart Product Review Sentiment Analyzer")

# Input text
user_input = st.text_area("Enter your product review:", height=150)

# Prediction
if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)
        sentiment = label_encoder.inverse_transform(pred)[0]

        # Output
        st.success(f"Predicted Sentiment: **{sentiment.capitalize()}**")

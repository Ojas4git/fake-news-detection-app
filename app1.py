import streamlit as st
import pickle
import re

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', '', text)
    text = text.strip()
    return text

# UI
st.title("📰 Fake News Detection App")
st.write("Detect whether a news article is Fake or Real using Machine Learning")

# Input box
input_text = st.text_area("Enter News Text:")

# Button
if st.button("Predict"):
    if input_text:
        # Clean input
        cleaned_input = clean_text(input_text)

        # Vectorize
        vectorized_text = vectorizer.transform([cleaned_input])

        # Predict
        prediction = model.predict(vectorized_text)
        proba = model.predict_proba(vectorized_text)

        confidence = max(proba[0])

        # Output
        if prediction[0] == 0:
            st.error(f"🚨 Fake News (Confidence: {confidence:.2f})")
        else:
            st.success(f"✅ Real News (Confidence: {confidence:.2f})")
    else:
        st.warning("Please enter some text")
# app_streamlit.py
import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model and vectorizer
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# App title and styling
st.set_page_config(page_title="Fake News Detection", page_icon="🧠", layout="centered")

st.markdown("""
    <h1 style='text-align: center; color: #FF4081;'>🧠 Fake News Detection System</h1>
    <p style='text-align: center; color: gray;'>Enter a news article or headline to check whether it's real or fake!</p>
""", unsafe_allow_html=True)

# Text input
user_input = st.text_area("📰 Paste the News Article Here:", height=200, placeholder="Enter the news content...")

# Predict button
if st.button("Detect"):
    if not user_input.strip():
        st.warning("⚠️ Please enter a news article before clicking Detect.")
    else:
        transformed_input = vectorizer.transform([user_input])
        prediction = model.predict(transformed_input)[0]
        
        if prediction == 1:
            st.success("✅ This News is **REAL**!")
        else:
            st.error("🚨 This News is **FAKE!**")

# Footer
st.markdown("""
<hr>
<p style='text-align: center; color: #999;'>Developed with ❤️ by Maithili</p>
""", unsafe_allow_html=True)


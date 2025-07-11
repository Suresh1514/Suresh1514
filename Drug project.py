# Updated Drug Review Streamlit App Using Reviews for Disease Prediction

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import re
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

st.set_page_config(page_title="Drug Reviews Analysis & Recommendation", layout="wide")

@st.cache_data
def load_data():
    try:
        data = pd.read_excel("drugsCom_raw.xlsx")
    except:
        st.error("Dataset not found and cannot continue.")
        return pd.DataFrame()

    target_conditions = ['Depression', 'High Blood Pressure', 'Diabetes, Type 2']
    data = data[data['condition'].isin(target_conditions)]
    data.dropna(subset=['review', 'condition'], inplace=True)
    return data

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

def analyze_sentiment(text):
    analysis = TextBlob(str(text))
    polarity = analysis.sentiment.polarity
    if polarity > 0.2:
        return 'Positive'
    elif polarity < -0.2:
        return 'Negative'
    elif polarity > 0:
        return 'Slightly Positive'
    elif polarity < 0:
        return 'Slightly Negative'
    else:
        return 'Neutral'

def train_model(data):
    data['processed_review'] = data['review'].apply(preprocess_text)
    vectorizer = TfidfVectorizer(max_features=2000)
    X = vectorizer.fit_transform(data['processed_review'])
    le = LabelEncoder()
    y = le.fit_transform(data['condition'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    st.sidebar.success(f"Model accuracy: {accuracy:.2f}")
    return vectorizer, le, model

def recommend_drugs_from_review(review_text, data, vectorizer, model, encoder):
    processed = preprocess_text(review_text)
    vector = vectorizer.transform([processed])
    disease_encoded = model.predict(vector)
    disease = encoder.inverse_transform(disease_encoded)[0]

    st.write(f"Processed review: {processed}")
    st.write(f"Predicted condition: {disease}")

    drugs = data[data['condition'] == disease]
    top = drugs.groupby('drugName')['rating'].mean().sort_values(ascending=False).head(3)
    return disease, top

# --- Main App ---
data = load_data()

if data.empty:
    st.stop()

vectorizer, encoder, model = train_model(data)

st.title("💊 Drug Review-Based Disease Prediction and Recommendation")

review_input = st.text_area("Paste your drug review or experience:", "This drug helped me sleep and reduced my sadness")

if st.button("Predict Disease & Recommend Drug"):
    if review_input.strip():
        condition, top_drugs = recommend_drugs_from_review(review_input, data, vectorizer, model, encoder)
        st.success(f"Predicted Condition: {condition}")
        st.subheader("Top Recommended Drugs:")
        for drug, rating in top_drugs.items():
            st.markdown(f"- **{drug}**: Avg Rating {rating:.1f}")
    else:
        st.warning("Please enter a review to analyze.")

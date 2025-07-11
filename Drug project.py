# Updated Drug Review Streamlit App with Bug Fix for Biased Prediction

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
        # More balanced synthetic data
        sample_data = {
            'drugName': ['Prozac', 'Lisinopril', 'Metformin', 'Zoloft', 'Amlodipine',
                         'Lexapro', 'Hydrochlorothiazide', 'Insulin Glargine', 'Sertraline',
                         'Losartan', 'Empagliflozin', 'Fluoxetine', 'Atenolol', 'Glipizide'],
            'condition': ['Depression', 'High Blood Pressure', 'Diabetes, Type 2',
                          'Depression', 'High Blood Pressure', 'Depression',
                          'High Blood Pressure', 'Diabetes, Type 2', 'Depression',
                          'High Blood Pressure', 'Diabetes, Type 2', 'Depression',
                          'High Blood Pressure', 'Diabetes, Type 2'],
            'review': [
                "This medication changed my life!",
                "Lowered BP but caused dizziness",
                "Controlled my sugar levels well",
                "Didn't help, felt worse",
                "Works well with minor side effects",
                "Improved slowly over weeks",
                "Effective but frequent urination",
                "Essential for diabetes",
                "Helped with anxiety and depression",
                "Effective, no side effects",
                "Controlled sugar effectively",
                "Effective after some time",
                "Mild improvement",
                "Reduced my A1C"
            ],
            'rating': [9, 7, 6, 3, 8, 8, 6, 9, 7, 8, 8, 7, 6, 8],
            'usefulCount': [45, 32, 28, 19, 37, 42, 25, 50, 33, 40, 38, 29, 22, 35]
        }
        data = pd.DataFrame(sample_data)

    target_conditions = ['Depression', 'High Blood Pressure', 'Diabetes, Type 2']
    data = data[data['condition'].isin(target_conditions)]

    condition_symptoms = {
        'Depression': 'sadness, hopelessness, loss of interest, insomnia, fatigue',
        'High Blood Pressure': 'high bp, headache, dizziness, blurred vision, nosebleed',
        'Diabetes, Type 2': 'thirst, frequent urination, fatigue, blurry vision, slow healing'
    }
    data['symptoms'] = data['condition'].map(condition_symptoms)
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
    data['processed_symptoms'] = data['symptoms'].apply(preprocess_text)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['processed_symptoms'])
    le = LabelEncoder()
    y = le.fit_transform(data['condition'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    st.sidebar.success(f"Model accuracy: {accuracy:.2f}")
    return vectorizer, le, model

def recommend_drugs(symptoms, data, vectorizer, model, encoder):
    processed = preprocess_text(symptoms)
    vector = vectorizer.transform([processed])
    disease_encoded = model.predict(vector)
    disease = encoder.inverse_transform(disease_encoded)[0]

    st.write(f"Processed symptoms: {processed}")
    st.write(f"Predicted condition: {disease}")

    drugs = data[data['condition'] == disease]
    top = drugs.groupby('drugName')['rating'].mean().sort_values(ascending=False).head(3)
    return disease, top

# --- Main App ---
data = load_data()
vectorizer, encoder, model = train_model(data)

st.title("💊 Drug Recommendation System")

symptoms_input = st.text_area("Enter your symptoms (comma-separated):", "fatigue, insomnia, sadness")

if st.button("Predict & Recommend"):
    if symptoms_input.strip():
        condition, top_drugs = recommend_drugs(symptoms_input, data, vectorizer, model, encoder)
        st.success(f"Predicted Condition: {condition}")
        st.subheader("Top Recommended Drugs:")
        for drug, rating in top_drugs.items():
            st.markdown(f"- **{drug}**: Avg Rating {rating:.1f}")
    else:
        st.warning("Please enter symptoms.")

# Streamlit App: Predict Disease & Recommend Drug using Patient Reviews

import streamlit as st
import pandas as pd
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

st.set_page_config(page_title="Review-Based Disease Prediction & Drug Recommendation", layout="wide")

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

@st.cache_data
def load_data():
    try:
        # Create a sample dataset directly in the code
        sample_data = {
            'drugName': ['Prozac', 'Lisinopril', 'Metformin', 'Zoloft', 'Atenolol', 'Glipizide',
                        'Lexapro', 'Hydrochlorothiazide', 'Januvia', 'Paxil'],
            'condition': ['Depression', 'High Blood Pressure', 'Diabetes, Type 2', 
                          'Depression', 'High Blood Pressure', 'Diabetes, Type 2',
                          'Depression', 'High Blood Pressure', 'Diabetes, Type 2', 'Depression'],
            'review': [
                'This medication helped with my depression symptoms',
                'Effective at controlling my blood pressure',
                'Helped lower my blood sugar levels',
                'Made me feel better but had some side effects',
                'Good for blood pressure but caused dizziness',
                'Worked well for my diabetes management',
                'Improved my mood significantly',
                'Reduced my blood pressure effectively',
                'Helped control my blood sugar with no side effects',
                'Took time to work but eventually helped'
            ],
            'rating': [9.0, 8.5, 8.0, 7.5, 7.0, 8.5, 9.0, 8.0, 8.5, 7.0]
        }
        
        data = pd.DataFrame(sample_data)
        conditions = ['Depression', 'High Blood Pressure', 'Diabetes, Type 2']
        filtered = data[data['condition'].isin(conditions)].copy()
        return filtered
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return pd.DataFrame()

def train_model(data):
    data['processed_review'] = data['review'].apply(preprocess_text)
    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1,2))
    X = vectorizer.fit_transform(data['processed_review'])
    le = LabelEncoder()
    y = le.fit_transform(data['condition'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    st.sidebar.success(f"Model Accuracy: {acc:.2f}")
    return vectorizer, le, model

def predict_and_recommend(review_text, data, vectorizer, model, encoder):
    processed = preprocess_text(review_text)
    vector = vectorizer.transform([processed])
    condition = encoder.inverse_transform(model.predict(vector))[0]
    top_drugs = (
        data[data['condition'] == condition]
        .groupby('drugName')['rating']
        .mean()
        .sort_values(ascending=False)
        .head(3)
    )
    return condition, top_drugs

st.title("💊 Disease Prediction and Drug Recommendation from Review Text")

data = load_data()

if not data.empty:
    vectorizer, encoder, model = train_model(data)

    st.markdown("### ✍️ Enter a Review Below")
    review_input = st.text_area("Example: 'This medicine helped lower my blood sugar and gave me energy.'")

    if st.button("Predict & Recommend"):
        if review_input.strip():
            condition, recommendations = predict_and_recommend(review_input, data, vectorizer, model, encoder)
            st.success(f"🩺 Predicted Condition: **{condition}**")
            st.markdown("### 💊 Top Recommended Drugs:")
            for drug, rating in recommendations.items():
                st.markdown(f"- **{drug}** — Avg Rating: {rating:.1f}")
        else:
            st.warning("Please enter a review to proceed.")
else:
    st.warning("Failed to load dataset. Please try again later.")

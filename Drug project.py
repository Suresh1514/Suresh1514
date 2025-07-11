# Streamlit App: Predict Disease & Recommend Drug using Patient Reviews

import streamlit as st
import pandas as pd
import re
import numpy as np
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

st.set_page_config(page_title="Review-Based Disease Prediction & Drug Recommendation", layout="wide")

def preprocess_text(text):
    try:
        if not isinstance(text, str) or not text.strip():
            return ""
        
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        words = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        words = [word for word in words if len(word) > 2]
        return ' '.join(words)
    except Exception as e:
        st.error(f"Text preprocessing error: {e}")
        return ""

def create_sample_dataset():
    conditions = ['Depression', 'High Blood Pressure', 'Diabetes, Type 2']
    
    depression_reviews = [
        "This medication helped with my depression and anxiety",
        "I feel much better after taking this for my depressive symptoms",
        "Effective for treating major depressive disorder",
        "Improved my mood significantly within weeks",
        "Reduced my depressive thoughts and improved sleep"
    ]
    
    bp_reviews = [
        "Effectively controls my high blood pressure",
        "Lowered my BP to normal levels with no side effects",
        "Great for hypertension management",
        "Works well to maintain healthy blood pressure",
        "My doctor is pleased with my BP numbers now"
    ]
    
    diabetes_reviews = [
        "Helped lower my blood sugar levels effectively",
        "Good for managing type 2 diabetes",
        "Reduced my A1C significantly",
        "Controls my glucose levels throughout the day",
        "Essential part of my diabetes treatment plan"
    ]
    
    drugs = {
        'Depression': ['Prozac', 'Zoloft', 'Lexapro', 'Paxil', 'Effexor'],
        'High Blood Pressure': ['Lisinopril', 'Atenolol', 'Losartan', 'Amlodipine', 'Valsartan'],
        'Diabetes, Type 2': ['Metformin', 'Glipizide', 'Januvia', 'Actos', 'Victoza']
    }
    
    data = []
    for condition in conditions:
        reviews = depression_reviews if condition == 'Depression' else bp_reviews if condition == 'High Blood Pressure' else diabetes_reviews
        for i, drug in enumerate(drugs[condition]):
            for j in range(5):
                rating = np.random.uniform(8.0, 9.5)
                data.append({
                    'drugName': drug,
                    'condition': condition,
                    'review': reviews[j],
                    'rating': round(rating, 1)
                })
    
    return pd.DataFrame(data)

@st.cache_data
def load_data():
    try:
        return create_sample_dataset()
    except Exception as e:
        st.error(f"Failed to create dataset: {e}")
        return pd.DataFrame()

def train_model(data):
    try:
        if data.empty:
            st.error("No data available for training")
            return None, None, None
        
        data['processed_review'] = data['review'].apply(preprocess_text)
        data = data[data['processed_review'].str.len() > 0]
        
        if len(data) == 0:
            st.error("No valid reviews after preprocessing")
            return None, None, None
        
        vectorizer = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        X = vectorizer.fit_transform(data['processed_review'])
        le = LabelEncoder()
        y = le.fit_transform(data['condition'])
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        st.success(f"Model trained successfully (Accuracy: {acc:.2f})")
        return vectorizer, le, model
        
    except Exception as e:
        st.error(f"Model training failed: {e}")
        return None, None, None

def main():
    st.title("💊 Disease Prediction and Drug Recommendation from Review Text")
    
    data = load_data()
    
    if not data.empty:
        vectorizer, encoder, model = train_model(data)
        
        if model:
            st.markdown("### ✍️ Enter a Patient Review Below")
            review_input = st.text_area(
                "Example: 'This medicine helped lower my blood sugar and gave me energy.'",
                height=150
            )
            
            if st.button("Predict & Recommend"):
                if review_input.strip():
                    processed = preprocess_text(review_input)
                    if processed:
                        vector = vectorizer.transform([processed])
                        prediction = model.predict(vector)
                        condition = encoder.inverse_transform(prediction)[0]
                        
                        st.success(f"🩺 Predicted Condition: **{condition}**")
                        
                        recommendations = (
                            data[data['condition'] == condition]
                            .groupby('drugName')['rating']
                            .mean()
                            .sort_values(ascending=False)
                            .head(3)
                        )
                        
                        if not recommendations.empty:
                            st.markdown("### 💊 Top Recommended Drugs:")
                            for drug, rating in recommendations.items():
                                st.markdown(f"- **{drug}** — Avg Rating: {rating:.1f}")
                    else:
                        st.warning("Please enter a valid review")
                else:
                    st.warning("Please enter a review to proceed")

if __name__ == "__main__":
    main()

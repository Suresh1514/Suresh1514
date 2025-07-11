# Streamlit App: Predict Disease & Recommend Drug using Patient Reviews

import streamlit as st
import pandas as pd
import re
import numpy as np
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

st.set_page_config(page_title="Review-Based Disease Prediction & Drug Recommendation", layout="wide")

# Enhanced text preprocessing
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenization
    words = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    # Remove short words (length < 3)
    words = [word for word in words if len(word) > 2]
    
    return ' '.join(words)

@st.cache_data
def load_data():
    try:
        # Expanded sample dataset with more examples
        sample_data = {
            'drugName': ['Prozac', 'Lisinopril', 'Metformin', 'Zoloft', 'Atenolol', 'Glipizide',
                        'Lexapro', 'Hydrochlorothiazide', 'Januvia', 'Paxil', 'Effexor', 'Amlodipine',
                        'Glucophage', 'Celexa', 'Losartan', 'Actos', 'Wellbutrin', 'Valsartan',
                        'Victoza', 'Cymbalta'],
            'condition': ['Depression', 'High Blood Pressure', 'Diabetes, Type 2', 
                          'Depression', 'High Blood Pressure', 'Diabetes, Type 2',
                          'Depression', 'High Blood Pressure', 'Diabetes, Type 2', 'Depression',
                          'Depression', 'High Blood Pressure', 'Diabetes, Type 2', 'Depression',
                          'High Blood Pressure', 'Diabetes, Type 2', 'Depression', 'High Blood Pressure',
                          'Diabetes, Type 2', 'Depression'],
            'review': [
                'This medication significantly improved my mood and reduced my depressive symptoms',
                'Effectively controls my hypertension with minimal side effects',
                'Great for managing my type 2 diabetes and blood sugar levels',
                'Helped with my depression but caused some initial nausea',
                'Works well for blood pressure though sometimes makes me dizzy',
                'Excellent for glucose control in my type 2 diabetes',
                'My anxiety and depression have improved dramatically',
                'Perfect for my high blood pressure management',
                'Works wonders for my diabetes with no noticeable side effects',
                'Took several weeks but eventually helped my depressive symptoms',
                'Venlafaxine changed my life, depression is much better',
                'Amlodipine keeps my blood pressure perfectly controlled',
                'Metformin is essential for my type 2 diabetes treatment',
                'Citalopram helped stabilize my mood swings',
                'Losartan is excellent for my hypertension',
                'Pioglitazone helps control my blood sugar effectively',
                'Bupropion gave me energy and helped with depression',
                'Valsartan works better than previous blood pressure meds',
                'Liraglutide helped me lose weight and control diabetes',
                'Duloxetine works well for both my pain and depression'
            ],
            'rating': [9.5, 9.0, 9.2, 8.0, 8.5, 9.1, 9.3, 9.0, 9.4, 8.2, 
                      9.6, 9.1, 9.3, 8.8, 9.2, 8.9, 9.0, 9.3, 9.5, 9.1]
        }
        
        # Create multiple variations of each review to expand the dataset
        expanded_data = []
        for i in range(len(sample_data['drugName'])):
            for variation in range(5):  # Create 5 variations of each review
                new_review = sample_data['review'][i]
                if variation == 1:
                    new_review = new_review.replace('.', '!')
                elif variation == 2:
                    new_review = new_review + " I highly recommend it."
                elif variation == 3:
                    new_review = "After using this, " + new_review.lower()
                elif variation == 4:
                    new_review = new_review.replace("my", "our")
                
                expanded_data.append({
                    'drugName': sample_data['drugName'][i],
                    'condition': sample_data['condition'][i],
                    'review': new_review,
                    'rating': sample_data['rating'][i] - (variation * 0.1)  # Slight rating variation
                })
        
        data = pd.DataFrame(expanded_data)
        return data
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return pd.DataFrame()

def train_model(data):
    try:
        # Enhanced text processing
        data['processed_review'] = data['review'].apply(preprocess_text)
        
        # Sentiment analysis features
        data['sentiment'] = data['review'].apply(lambda x: TextBlob(x).sentiment.polarity)
        data['subjectivity'] = data['review'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
        
        # TF-IDF with optimized parameters
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        
        X_text = vectorizer.fit_transform(data['processed_review'])
        
        # Combine text features with sentiment features
        X_sentiment = data[['sentiment', 'subjectivity']].values
        X = np.hstack((X_text.toarray(), X_sentiment))
        
        # Encode labels
        le = LabelEncoder()
        y = le.fit_transform(data['condition'])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )
        
        # Handle class imbalance
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        
        # Optimized model with hyperparameter tuning
        model = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=le.classes_)
        
        st.sidebar.success(f"Model Accuracy: {acc:.2f}")
        st.sidebar.text("Classification Report:\n" + report)
        
        return vectorizer, le, model
    
    except Exception as e:
        st.error(f"Model training failed: {e}")
        return None, None, None

def predict_and_recommend(review_text, data, vectorizer, model, encoder):
    try:
        # Preprocess the input text
        processed = preprocess_text(review_text)
        
        # Get sentiment features
        sentiment = TextBlob(review_text).sentiment.polarity
        subjectivity = TextBlob(review_text).sentiment.subjectivity
        
        # Vectorize the text
        text_vector = vectorizer.transform([processed]).toarray()
        
        # Combine with sentiment features
        features = np.hstack((text_vector, [[sentiment, subjectivity]]))
        
        # Make prediction
        condition = encoder.inverse_transform(model.predict(features))[0]
        
        # Get top recommendations
        top_drugs = (
            data[data['condition'] == condition]
            .groupby('drugName')['rating']
            .mean()
            .sort_values(ascending=False)
            .head(5)  # Show top 5 recommendations
        )
        
        return condition, top_drugs
    
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None, None

st.title("💊 Disease Prediction and Drug Recommendation from Review Text")

data = load_data()

if not data.empty:
    vectorizer, encoder, model = train_model(data)

    st.markdown("### ✍️ Enter a Review Below")
    review_input = st.text_area("Example: 'This medicine helped lower my blood sugar and gave me energy.'",
                              height=150)

    if st.button("Predict & Recommend"):
        if review_input.strip():
            condition, recommendations = predict_and_recommend(review_input, data, vectorizer, model, encoder)
            if condition:
                st.success(f"🩺 Predicted Condition: **{condition}**")
                st.markdown("### 💊 Top Recommended Drugs:")
                for drug, rating in recommendations.items():
                    st.markdown(f"- **{drug}** — Avg Rating: {rating:.1f}")
                
                # Show similar reviews from training data
                st.markdown("### 📝 Similar Reviews from Training Data:")
                similar_reviews = data[data['condition'] == condition].sample(3)
                for idx, row in similar_reviews.iterrows():
                    st.markdown(f"- \"{row['review']}\" (Rating: {row['rating']})")
        else:
            st.warning("Please enter a review to proceed.")
else:
    st.warning("Failed to load dataset. Please try again later.")

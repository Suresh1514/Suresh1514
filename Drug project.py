import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
import streamlit as st
import os

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Set page config
st.set_page_config(
    page_title="Drug Recommendation System",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load dataset from Excel file
@st.cache_data
def load_data():
    try:
        # Load the Excel file
        df = pd.read_excel('drugsCom_raw.xlsx', engine='openpyxl')
        
        # Standardize column names
        column_mapping = {
            'drugName': 'drugName',
            'condition': 'condition',
            'review': 'review',
            'rating': 'rating',
            'date': 'date',
            'usefulCount': 'usefulCount'
        }
        
        # Rename columns to standard names
        df.columns = [column_mapping.get(col, col) for col in df.columns]
        
        # Filter for our target conditions
        target_conditions = ['Depression', 'High Blood Pressure', 'Diabetes, Type 2']
        df = df[df['condition'].isin(target_conditions)]
        
        return df
    
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

# Text preprocessing functions
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\d+', '', text)
    return ' '.join(text.split())

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in text.split() if word not in stop_words])

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

# Train and cache the model
@st.cache_resource
def train_model(df):
    # Apply preprocessing
    df['cleaned_review'] = df['review'].apply(preprocess_text).apply(remove_stopwords).apply(lemmatize_text)
    
    # Prepare data for modeling
    X = df['cleaned_review']
    y = df['condition']
    
    # Vectorize text data
    vectorizer = TfidfVectorizer(max_features=5000)
    X_vec = vectorizer.fit_transform(X)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LogisticRegression(max_iter=1000, multi_class='multinomial')
    model.fit(X_train, y_train)
    
    return model, vectorizer, df

# Main application
def main():
    st.title("💊 Drug Recommendation System")
    st.markdown("""
    Analyze patient reviews to recommend medications for:
    - **Depression**
    - **High Blood Pressure** 
    - **Diabetes, Type 2**
    """)
    
    # Load data
    df = load_data()
    
    if df is None:
        st.error("Failed to load dataset. Please ensure 'drugsCom_raw.xlsx' exists in the same directory.")
        return
    
    # Train model
    model, vectorizer, processed_df = train_model(df)
    
    # User input
    st.subheader("Describe Your Symptoms or Medication Experience")
    user_review = st.text_area(
        "Enter your text here:", 
        height=150,
        placeholder="e.g., 'I've been feeling very sad and hopeless lately...'"
    )
    
    if st.button("Get Recommendation", type="primary"):
        if user_review:
            with st.spinner("Analyzing your input..."):
                # Preprocess input
                processed_input = lemmatize_text(remove_stopwords(preprocess_text(user_review)))
                vec = vectorizer.transform([processed_input])
                
                # Predict condition
                prediction = model.predict(vec)[0]
                
                st.markdown("---")
                st.subheader("Results")
                
                # Display prediction with icon
                condition_icons = {
                    'Depression': '😔',
                    'High Blood Pressure': '🩺', 
                    'Diabetes, Type 2': '🩸'
                }
                st.success(f"**Predicted Condition:** {condition_icons[prediction]} {prediction}")
                
                # Get top recommended drugs
                top_drugs = (
                    processed_df[processed_df['condition'] == prediction]
                    .groupby('drugName')
                    .agg({'rating': 'mean', 'usefulCount': 'sum'})
                    .sort_values(['rating', 'usefulCount'], ascending=False)
                    .head(3)
                )
                
                if not top_drugs.empty:
                    st.subheader("💊 Recommended Medications")
                    
                    # Display as columns
                    cols = st.columns(3)
                    for idx, (drug, row) in enumerate(top_drugs.iterrows()):
                        with cols[idx]:
                            st.metric(
                                label=drug,
                                value=f"{row['rating']:.1f} ★",
                                help=f"Based on {int(row['usefulCount'])} user ratings"
                            )
                            # Show a sample review
                            sample_review = processed_df[
                                (processed_df['condition'] == prediction) & 
                                (processed_df['drugName'] == drug)
                            ].iloc[0]['review']
                            st.caption(f'"{sample_review[:60]}..."')
                else:
                    st.warning("No recommendations available for this condition")
        else:
            st.warning("Please describe your symptoms or medication experience")
    
    # Dataset info in sidebar
    with st.sidebar:
        st.header("Dataset Info")
        st.markdown(f"""
        - **Total entries:** {len(df)}
        - **Conditions:**
          - Depression: {len(df[df['condition'] == 'Depression'])}
          - High BP: {len(df[df['condition'] == 'High Blood Pressure'])} 
          - Diabetes Type 2: {len(df[df['condition'] == 'Diabetes, Type 2'])}
        """)
        
        if st.checkbox("Show raw data"):
            st.dataframe(df)

if __name__ == '__main__':
    main()

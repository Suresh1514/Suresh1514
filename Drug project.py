# Import required libraries
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
from io import BytesIO

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Set page config for Streamlit
st.set_page_config(
    page_title="Drug Recommendation System",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create sample data function
def create_sample_data():
    """Create sample data as fallback"""
    data = {
        'drugName': ['Prozac', 'Lisinopril', 'Metformin', 'Prozac', 'Metformin'],
        'condition': ['Depression', 'High Blood Pressure', 'Diabetes, Type 2', 'Depression', 'Diabetes, Type 2'],
        'review': [
            'This medication helped with my depression symptoms',
            'Effective for controlling my blood pressure',
            'Great for managing my type 2 diabetes',
            'Did not help with my depression at all',
            'Works well for diabetes control with minimal side effects'
        ],
        'rating': [8, 9, 7, 2, 8],
        'date': ['2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01', '2020-05-01'],
        'usefulCount': [10, 15, 12, 3, 20]
    }
    return pd.DataFrame(data)

# Data loading function with multiple fallbacks
@st.cache_data
def load_data(uploaded_file=None):
    """Load data with multiple fallback options"""
    df = None
    
    # If user uploaded a file
    if uploaded_file is not None:
        try:
            # Try to read as Excel
            try:
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            except:
                # Try to read as CSV
                try:
                    uploaded_file.seek(0)  # Reset file pointer
                    df = pd.read_csv(uploaded_file)
                except:
                    # Try to read as TSV
                    try:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, sep='\t')
                    except Exception as e:
                        st.error(f"Could not read uploaded file: {str(e)}")
        
        except Exception as e:
            st.error(f"Error processing uploaded file: {str(e)}")
    
    # If no file uploaded or loading failed, use sample data
    if df is None:
        st.info("Using sample data as no valid file was provided or could be loaded.")
        return create_sample_data()
    
    # Standardize column names (handle different capitalization/spacing)
    column_mapping = {
        'drugname': 'drugName',
        'drug_name': 'drugName',
        'conditions': 'condition',
        'ratings': 'rating',
        'usefulcount': 'usefulCount'
    }
    
    df.columns = [column_mapping.get(col.lower(), col) for col in df.columns]
    
    # Verify required columns
    required_columns = ['drugName', 'condition', 'review', 'rating', 'usefulCount']
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        st.warning(f"Missing columns: {', '.join(missing_cols)}. Using sample data instead.")
        return create_sample_data()
    
    return df

# Text preprocessing functions
def preprocess_text(text):
    """Clean and preprocess text"""
    text = str(text).lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\d+', '', text)
    return ' '.join(text.split())

def remove_stopwords(text):
    """Remove stopwords from text"""
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in text.split() if word not in stop_words])

def lemmatize_text(text):
    """Lemmatize words in text"""
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

# Train model function
def train_model(df):
    """Train and return the classification model"""
    # Focus on target conditions
    target_conditions = ['Depression', 'High Blood Pressure', 'Diabetes, Type 2']
    df = df[df['condition'].isin(target_conditions)]
    
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

# Main application function
def main():
    st.title("💊 Drug Recommendation System")
    st.markdown("""
    This system analyzes patient reviews to:
    - Classify medical conditions
    - Recommend appropriate medications
    """)
    
    # Sidebar for file upload and info
    with st.sidebar:
        st.header("Upload Your Data")
        uploaded_file = st.file_uploader(
            "Upload drug reviews (Excel, CSV, or TSV)",
            type=['xlsx', 'csv', 'tsv'],
            help="File should contain columns: drugName, condition, review, rating, usefulCount"
        )
        
        st.markdown("---")
        st.markdown("""
        **Target Conditions:**
        - Depression
        - High Blood Pressure
        - Diabetes, Type 2
        """)
        
        st.markdown("---")
        st.markdown("""
        **How it works:**
        1. Enter your symptoms or medication experience
        2. System analyzes the text
        3. Predicts the most likely condition
        4. Recommends top medications
        """)
    
    # Load data
    df = load_data(uploaded_file)
    
    # Train model (cached for performance)
    @st.cache_resource
    def get_model():
        return train_model(df)
    
    model, vectorizer, processed_df = get_model()
    
    # Main interface
    st.subheader("Enter Your Symptoms or Medication Experience")
    user_review = st.text_area(
        "Describe how you're feeling or your experience with a medication:",
        height=150,
        placeholder="e.g., 'I've been feeling very down and hopeless lately...'"
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
                st.subheader("Analysis Results")
                
                if prediction in ['Depression', 'High Blood Pressure', 'Diabetes, Type 2']:
                    # Display prediction
                    condition_icons = {
                        'Depression': '😔',
                        'High Blood Pressure': '🩺',
                        'Diabetes, Type 2': '🩸'
                    }
                    st.success(f"""
                    **Predicted Condition:** {condition_icons[prediction]} {prediction}
                    """)
                    
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
                        
                        cols = st.columns(3)
                        for idx, (drug, row) in enumerate(top_drugs.iterrows()):
                            with cols[idx]:
                                st.metric(
                                    label=drug,
                                    value=f"{row['rating']:.1f} ★",
                                    help=f"Based on {int(row['usefulCount'])} user ratings"
                                )
                                st.caption(f"Average rating from patient reviews")
                    else:
                        st.warning("No medication recommendations available for this condition")
                else:
                    st.warning("""
                    The system couldn't match your input to our target conditions.
                    Please describe symptoms related to:
                    - Depression
                    - High Blood Pressure
                    - Diabetes Type 2
                    """)
        else:
            st.warning("Please describe your symptoms or medication experience")

    # Show sample data if in debug mode
    if st.sidebar.checkbox("Show sample data (debug)"):
        st.subheader("Sample Data Preview")
        st.dataframe(df.head())

if __name__ == '__main__':
    main()

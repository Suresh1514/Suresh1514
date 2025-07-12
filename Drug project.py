import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import re
from textblob import TextBlob
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import joblib
import os

# Set page configuration
st.set_page_config(
    page_title="Drug Reviews Analysis",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data function with caching
@st.cache_data
def load_data():
    try:
        data = pd.read_excel("drugsCom_raw.xlsx") 
    except Exception as e:
        st.warning(f"Using sample data as the real dataset wasn't found. Error: {e}")
        # Enhanced sample data with more entries for each condition
        sample_data = {
            'drugName': ['Prozac', 'Lisinopril', 'Metformin', 'Zoloft', 'Amlodipine',
                        'Lexapro', 'Hydrochlorothiazide', 'Insulin Glargine', 'Sertraline',
                        'Losartan', 'Empagliflozin', 'Fluoxetine', 'Atenolol', 'Glipizide'],
            'condition': ['Depression', 'High Blood Pressure', 'Diabetes, Type 2', 
                        'Depression', 'High Blood Pressure', 'Depression',
                        'High Blood Pressure', 'Diabetes, Type 2', 'Depression',
                        'High Blood Pressure', 'Diabetes, Type 2', 'Depression',
                        'High Blood Pressure', 'Diabetes, Type 2'],
            'review': ["This medication changed my life!", 
                      "Helped lower my blood pressure but caused dizziness",
                      "Effective for sugar control but upset my stomach",
                      "Didn't work for me, made me more anxious",
                      "Works well with minimal side effects",
                      "Gradual improvement over several weeks",
                      "Good for blood pressure but frequent urination",
                      "Essential for my diabetes management",
                      "Helped with my anxiety and depression",
                      "Effective with no side effects for me",
                      "Great for blood sugar control",
                      "Took time to work but effective now",
                      "Mild effect on my blood pressure",
                      "Helped reduce my A1C levels"],
            'rating': [9, 7, 6, 3, 8, 8, 6, 9, 7, 8, 8, 7, 6, 8],
            'date': pd.to_datetime(['2020-01-15', '2021-03-22', '2022-05-10', 
                                  '2020-11-30', '2021-07-14', '2021-02-18',
                                  '2022-01-05', '2020-09-12', '2021-11-22',
                                  '2022-03-15', '2021-09-08', '2020-07-19',
                                  '2022-02-28', '2021-12-10']),
            'usefulCount': [45, 32, 28, 19, 37, 42, 25, 50, 33, 40, 38, 29, 22, 35]
        }
        data = pd.DataFrame(sample_data)
    
    # Filter for target conditions
    target_conditions = ['Depression', 'High Blood Pressure', 'Diabetes, Type 2']
    data = data[data['condition'].isin(target_conditions)]
    
    return data

# Enhanced sentiment analysis function
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

# Train or load drug prediction model
@st.cache_resource
def get_drug_prediction_model(data):
    model_path = "drug_prediction_model.joblib"
    
    # Check if model exists
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        return model
    
    # Prepare data for modeling
    df_model = data[['review', 'drugName']].copy()
    df_model = df_model.dropna()
    
    # Only train if we have enough data
    if len(df_model) < 10 or df_model['drugName'].nunique() < 2:
        return None
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df_model['review'], 
        df_model['drugName'], 
        test_size=0.2, 
        random_state=42
    )
    
    # Create pipeline
    model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Train model
    model.fit(X_train, y_train)
    
    # Save model
    joblib.dump(model, model_path)
    
    return model

# Function to predict drug based on review text
def predict_drug(model, review_text, data, top_n=3):
    if model is None or not review_text:
        return None
    
    try:
        # Get probabilities for all drugs
        probabilities = model.predict_proba([review_text])[0]
        drugs = model.classes_
        
        # Create dataframe with drugs and their probabilities
        results = pd.DataFrame({
            'Drug': drugs,
            'Probability': probabilities
        }).sort_values('Probability', ascending=False).head(top_n)
        
        # Add additional information from original data
        drug_info = data[['drugName', 'condition']].drop_duplicates()
        results = results.merge(drug_info, left_on='Drug', right_on='drugName', how='left')
        results = results.drop(columns=['drugName'])
        
        return results
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# Load data
data = load_data()

# Get prediction model
prediction_model = get_drug_prediction_model(data)

# Title and description
st.title("💊 Advanced Drug Reviews Analysis Dashboard")
st.markdown("""
This dashboard provides an in-depth analysis of patient reviews for medications treating **Depression**, 
**High Blood Pressure**, and **Diabetes (Type 2)**. Explore the data through interactive visualizations 
and comprehensive sentiment analysis.
""")

# Sidebar filters
st.sidebar.header("Filter Data")
selected_conditions = st.sidebar.multiselect(
    "Select conditions to analyze",
    options=data['condition'].unique(),
    default=data['condition'].unique().tolist()
)

min_rating, max_rating = st.sidebar.slider(
    "Select rating range",
    min_value=1,
    max_value=10,
    value=(1, 10)
)

# Date range filter
min_date = data['date'].min().to_pydatetime()
max_date = data['date'].max().to_pydatetime()
selected_dates = st.sidebar.date_input(
    "Select date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Filter data based on selections
filtered_data = data[
    (data['condition'].isin(selected_conditions)) & 
    (data['rating'] >= min_rating) & 
    (data['rating'] <= max_rating) &
    (data['date'] >= pd.to_datetime(selected_dates[0])) &
    (data['date'] <= pd.to_datetime(selected_dates[1]))
]

# Add sentiment analysis
if not filtered_data.empty:
    filtered_data['sentiment'] = filtered_data['review'].apply(analyze_sentiment)

# Main content - Added new tab for drug prediction
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Overview", "Rating Analysis", "Sentiment Analysis", 
    "Condition Comparison", "Word Cloud", "Data Explorer", "Drug Prediction"
])

# Previous tabs remain the same...

with tab7:
    st.header("Drug Prediction from Symptoms/Reviews")
    
    if prediction_model is None:
        st.warning("Drug prediction model could not be trained (insufficient data). Using sample data.")
        # Sample prediction results for demonstration
        sample_review = st.text_area("Enter your symptoms or medication experience:", 
                                   "I've been feeling depressed and anxious lately")
        
        if st.button("Predict Medication"):
            st.subheader("Sample Prediction Results (based on sample data)")
            sample_results = pd.DataFrame({
                'Drug': ['Prozac', 'Zoloft', 'Lexapro'],
                'Condition': ['Depression', 'Depression', 'Depression'],
                'Probability': [0.85, 0.75, 0.65]
            })
            st.dataframe(sample_results)
            
            st.markdown("""
            **Note:** This is a sample prediction. With more data, the model would analyze your symptoms 
            and predict the most likely medication based on similar reviews from patients.
            """)
    else:
        review_text = st.text_area("Enter your symptoms or medication experience:", 
                                 "I've been feeling depressed and anxious lately")
        
        if st.button("Predict Medication"):
            with st.spinner("Analyzing your symptoms..."):
                results = predict_drug(prediction_model, review_text, data)
                
                if results is not None and not results.empty:
                    st.subheader("Recommended Medications")
                    
                    # Display top prediction with more prominence
                    top_drug = results.iloc[0]
                    st.markdown(f"""
                    <div style="background-color:#e6f7ff; padding:15px; border-radius:10px; margin-bottom:20px;">
                        <h3 style="color:#1890ff;">Top Recommendation: {top_drug['Drug']}</h3>
                        <p><strong>Condition:</strong> {top_drug['condition']}</p>
                        <p><strong>Confidence:</strong> {top_drug['Probability']*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display all predictions in a table
                    st.write("Other possible medications:")
                    st.dataframe(results.style.format({'Probability': '{:.2%}'}))
                    
                    # Show reviews for the top recommended drug
                    st.subheader(f"Patient Reviews for {top_drug['Drug']}")
                    drug_reviews = data[data['drugName'] == top_drug['Drug']]['review'].head(5)
                    
                    for i, review in enumerate(drug_reviews, 1):
                        st.markdown(f"""
                        <div style="background-color:#f0f2f6; padding:10px; border-radius:5px; margin-bottom:10px;">
                        <b>Review {i}:</b> {review}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("Could not generate predictions. Please try different symptoms.")

# Add some space at the bottom
st.markdown("---")
st.markdown("""
**Note:** This application analyzes patient reviews for Depression, High Blood Pressure, and Diabetes (Type 2) medications.
The sentiment analysis categorizes reviews into five categories for more nuanced understanding.
The drug prediction model uses machine learning to suggest medications based on symptom descriptions.
""")

# Add requirements.txt for deployment
st.sidebar.markdown("""
**Deployment requirements.txt:**
streamlit
pandas
numpy
matplotlib
seaborn
wordcloud
textblob
python-dateutil
scikit-learn
joblib
""")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import re
from textblob import TextBlob
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Set page configuration
st.set_page_config(
    page_title="Drug Reviews Analysis & Recommendation",
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
    
    # Create synthetic symptoms if the column doesn't exist
    if 'symptoms' not in data.columns:
        condition_symptoms = {
            'Depression': 'sadness fatigue insomnia loss interest',
            'High Blood Pressure': 'headache dizziness blurred vision shortness breath',
            'Diabetes, Type 2': 'thirst frequent urination fatigue blurry vision'
        }
        data['symptoms'] = data['condition'].map(condition_symptoms)
    
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

# Text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Tokenize
    words = text.split()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Train disease classification model
def train_disease_model(data):
    # Preprocess symptoms
    data['processed_symptoms'] = data['symptoms'].apply(preprocess_text)
    
    # Vectorize symptoms
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(data['processed_symptoms'])
    
    # Encode conditions
    le = LabelEncoder()
    y = le.fit_transform(data['condition'])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.sidebar.info(f"Disease prediction model accuracy: {accuracy:.2f}")
    
    return vectorizer, le, model

# Enhanced drug recommendation function
def recommend_drugs(symptoms, data, vectorizer, disease_model, disease_encoder):
    # Preprocess input symptoms
    processed_symptoms = preprocess_text(symptoms)
    
    # Predict disease
    symptoms_vec = vectorizer.transform([processed_symptoms])
    disease_encoded = disease_model.predict(symptoms_vec)
    disease = disease_encoder.inverse_transform(disease_encoded)[0]
    
    # Debug output
    st.sidebar.markdown("**Model Insights:**")
    st.sidebar.write(f"Processed symptoms: {processed_symptoms}")
    st.sidebar.write(f"Predicted condition: {disease}")
    
    # Filter drugs for the predicted disease
    disease_drugs = data[data['condition'] == disease]
    
    if disease_drugs.empty:
        return None, disease, None, None
    
    # Calculate weighted score (70% rating, 30% usefulness)
    disease_drugs['weighted_score'] = (disease_drugs['rating'] * 0.7) + (disease_drugs['usefulCount'] * 0.3)
    
    # Get top drugs by weighted score
    top_drugs = disease_drugs.groupby('drugName').agg({
        'weighted_score': 'mean',
        'rating': 'mean',
        'usefulCount': 'mean',
        'review': 'count'
    }).sort_values(by=['weighted_score'], ascending=False).head(5)
    
    # Get sentiment distribution for top drugs
    drug_sentiments = {}
    for drug in top_drugs.index:
        drug_reviews = disease_drugs[disease_drugs['drugName'] == drug]
        drug_reviews['sentiment'] = drug_reviews['review'].apply(analyze_sentiment)
        sentiment_dist = drug_reviews['sentiment'].value_counts(normalize=True).to_dict()
        drug_sentiments[drug] = sentiment_dist
    
    return top_drugs, disease, drug_sentiments, disease_drugs

# Load data
data = load_data()

# Train disease classification model
vectorizer, disease_encoder, disease_model = train_disease_model(data)

# Title and description
st.title("💊 Advanced Drug Reviews Analysis Dashboard")
st.markdown("""
This dashboard provides an in-depth analysis of patient reviews for medications treating **Depression**, 
**High Blood Pressure**, and **Diabetes (Type 2)**. Explore the data through interactive visualizations 
and comprehensive sentiment analysis, and get personalized drug recommendations based on symptoms.
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

# Main content tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Overview", "Rating Analysis", "Sentiment Analysis", 
    "Condition Comparison", "Word Cloud", "Data Explorer", "Drug Recommendation"
])

# Drug Recommendation tab (updated)
with tab7:
    st.header("Drug Recommendation Based on Symptoms")
    
    # Display symptom-condition mapping for reference
    with st.expander("Symptom-Condition Reference"):
        st.markdown("""
        **Common Symptoms for Each Condition:**
        - **Depression:** sadness, fatigue, insomnia, loss of interest
        - **High Blood Pressure:** headache, dizziness, blurred vision, shortness of breath
        - **Diabetes, Type 2:** thirst, frequent urination, fatigue, blurry vision
        """)
    
    # User input for symptoms with better examples
    symptoms_input = st.text_area(
        "Enter your symptoms (comma or space separated):", 
        "headache dizziness",
        help="Example inputs: 'fatigue, sadness' for Depression, 'headache, blurred vision' for High BP, 'thirst, frequent urination' for Diabetes"
    )
    
    if st.button("Get Recommendation"):
        if symptoms_input.strip():
            # Get recommendations using the full dataset
            top_drugs, predicted_disease, drug_sentiments, disease_drugs = recommend_drugs(
                symptoms_input, data, vectorizer, disease_model, disease_encoder)
            
            if top_drugs is not None:
                st.success(f"Predicted Condition: {predicted_disease}")
                
                # Display top recommended drugs
                st.subheader("Top Recommended Drugs")
                for i, (drug, row) in enumerate(top_drugs.iterrows(), 1):
                    with st.expander(f"{i}. {drug} (Score: {row['weighted_score']:.1f}, Avg Rating: {row['rating']:.1f})"):
                        st.write(f"**Number of Reviews:** {row['review']}")
                        st.write(f"**Average Useful Count:** {row['usefulCount']:.1f}")
                        
                        # Display sentiment distribution
                        if drug in drug_sentiments:
                            st.write("**Sentiment Distribution:**")
                            sentiment_df = pd.DataFrame.from_dict(
                                drug_sentiments[drug], 
                                orient='index', 
                                columns=['Percentage']
                            ).sort_index()
                            sentiment_df['Percentage'] = (sentiment_df['Percentage'] * 100).round(1)
                            st.dataframe(sentiment_df.style.format({'Percentage': '{:.1f}%'}))
                        
                        # Display sample reviews
                        drug_reviews = disease_drugs[
                            (disease_drugs['drugName'] == drug) & 
                            (disease_drugs['review'].notna())
                        ]['review']
                        if not drug_reviews.empty:
                            st.write("**Sample Reviews:**")
                            for review in drug_reviews.sample(min(3, len(drug_reviews))):
                                st.write(f"- {review}")
                        else:
                            st.write("No reviews available for this drug.")
            else:
                st.warning(f"No drugs found for predicted condition: {predicted_disease}")
        else:
            st.warning("Please enter symptoms to get recommendations.")

# [Rest of your tabs (tab1 through tab6) remain unchanged...]

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
nltk
joblib
""")

# Add some space at the bottom
st.markdown("---")
st.markdown("""
**Note:** This application analyzes patient reviews for Depression, High Blood Pressure, and Diabetes (Type 2) medications.
The sentiment analysis categorizes reviews into five categories for more nuanced understanding.
The drug recommendation system suggests medications based on symptoms.
""")

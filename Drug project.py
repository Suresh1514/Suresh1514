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
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
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
    
    # Create synthetic symptoms with medical accuracy
    if 'symptoms' not in data.columns:
        condition_symptoms = {
            'Depression': 'low mood sadness hopelessness fatigue insomnia',
            'High Blood Pressure': 'headache dizziness nosebleeds shortness breath',
            'Diabetes, Type 2': 'thirst frequent urination hunger fatigue blurry vision'
        }
        data['symptoms'] = data['condition'].map(condition_symptoms)
    
    return data

# Enhanced sentiment analysis
def analyze_sentiment(text):
    analysis = TextBlob(str(text))
    polarity = analysis.sentiment.polarity
    
    if polarity > 0.3:
        return 'Strongly Positive'
    elif polarity > 0.1:
        return 'Positive'
    elif polarity > -0.1:
        return 'Neutral'
    elif polarity > -0.3:
        return 'Negative'
    else:
        return 'Strongly Negative'

# Improved text preprocessing
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    words = text.split()
    
    stop_words = set(stopwords.words('english'))
    medical_stopwords = {'drug', 'medication', 'doctor', 'prescribed', 'mg'}
    stop_words.update(medical_stopwords)
    
    words = [word for word in words if word not in stop_words]
    
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word, pos='v') for word in words]
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return ' '.join(words)

# Train disease classification model
def train_disease_model(data):
    data['processed_symptoms'] = data['symptoms'].apply(preprocess_text)
    
    vectorizer = TfidfVectorizer(max_features=1500, ngram_range=(1, 2))
    X = vectorizer.fit_transform(data['processed_symptoms'])
    
    le = LabelEncoder()
    y = le.fit_transform(data['condition'])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = RandomForestClassifier(
        n_estimators=150, 
        max_depth=10, 
        min_samples_split=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_)
    
    st.sidebar.info(f"Disease prediction model accuracy: {accuracy:.2f}")
    st.sidebar.text("Classification Report:\n" + report)
    
    return vectorizer, le, model

# Enhanced drug recommendation
def recommend_drugs(symptoms, data, vectorizer, disease_model, disease_encoder):
    processed_symptoms = preprocess_text(symptoms)
    symptoms_vec = vectorizer.transform([processed_symptoms])
    
    disease_probs = disease_model.predict_proba(symptoms_vec)[0]
    disease = disease_encoder.inverse_transform([np.argmax(disease_probs)])[0]
    confidence = np.max(disease_probs)
    
    st.sidebar.info(f"Prediction Confidence: {confidence:.2%}")
    
    disease_drugs = data[data['condition'] == disease]
    
    if disease_drugs.empty:
        return None, disease, None, None
    
    disease_drugs['score'] = (disease_drugs['rating'] * 0.6 + 
                             disease_drugs['usefulCount'] * 0.4)
    
    top_drugs = disease_drugs.groupby('drugName').agg({
        'score': 'mean',
        'rating': 'mean',
        'usefulCount': 'mean',
        'review': 'count'
    }).sort_values(by='score', ascending=False).head(5)
    
    drug_sentiments = {}
    for drug in top_drugs.index:
        drug_reviews = disease_drugs[disease_drugs['drugName'] == drug]
        if not drug_reviews.empty:
            drug_reviews['sentiment'] = drug_reviews['review'].apply(analyze_sentiment)
            sentiment_dist = drug_reviews['sentiment'].value_counts(normalize=True).to_dict()
            drug_sentiments[drug] = sentiment_dist
    
    return top_drugs, disease, drug_sentiments, disease_drugs

# Load data
data = load_data()

# Train disease classification model
vectorizer, disease_encoder, disease_model = train_disease_model(data)

# Title and description
st.title("💊 Patient Condition Classification & Drug Recommendation")
st.markdown("""
This dashboard analyzes patient reviews for medications treating:
- **Depression**
- **High Blood Pressure** 
- **Diabetes (Type 2)**

Key features:
- Classifies patient conditions from symptoms
- Analyzes drug effectiveness and side effects
- Provides personalized drug recommendations
""")

# Sidebar filters
st.sidebar.header("Data Filters")
selected_conditions = st.sidebar.multiselect(
    "Select conditions to analyze",
    options=data['condition'].unique(),
    default=data['condition'].unique().tolist()
)

min_rating, max_rating = st.sidebar.slider(
    "Select rating range (1-10):",
    min_value=1,
    max_value=10,
    value=(4, 8)
)

date_range = st.sidebar.date_input(
    "Select date range:",
    value=(data['date'].min(), data['date'].max()),
    min_value=data['date'].min(),
    max_value=data['date'].max()
)

# Filter data
filtered_data = data[
    (data['condition'].isin(selected_conditions)) & 
    (data['rating'].between(min_rating, max_rating)) &
    (data['date'].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])))
]

# Add sentiment analysis
if not filtered_data.empty:
    filtered_data['sentiment'] = filtered_data['review'].apply(analyze_sentiment)

# Main tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Overview", "Rating Analysis", "Sentiment Insights", 
    "Condition Comparison", "Review Analysis", "Data Explorer", 
    "Drug Recommendation"
])

# Overview Tab
with tab1:
    st.header("Data Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Reviews", len(filtered_data))
        st.metric("Unique Drugs", filtered_data['drugName'].nunique())
    with col2:
        st.metric("Average Rating", f"{filtered_data['rating'].mean():.1f}")
        st.metric("Most Reviewed Condition", filtered_data['condition'].value_counts().idxmax())
    with col3:
        st.metric("Most Common Sentiment", filtered_data['sentiment'].mode()[0])
        st.metric("Avg Useful Count", f"{filtered_data['usefulCount'].mean():.1f}")
    
    st.subheader("Top 10 Drugs by Review Count")
    top_drugs = filtered_data['drugName'].value_counts().nlargest(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(y=top_drugs.index, x=top_drugs.values, palette="viridis", ax=ax)
    ax.set_title('Top 10 Drugs by Review Count')
    ax.set_xlabel('Number of Reviews')
    st.pyplot(fig)
    
    st.subheader("Sample Data Preview")
    st.dataframe(filtered_data.head())

# Rating Analysis Tab
with tab2:
    st.header("Rating Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Rating Distribution by Condition")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=filtered_data, x='condition', y='rating', palette="Set2", ax=ax)
        ax.set_title('Rating Distribution by Condition')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Rating Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=filtered_data, x='rating', hue='condition', 
                    multiple='stack', bins=10, palette="Set2", ax=ax)
        ax.set_title('Stacked Rating Distribution')
        st.pyplot(fig)
    
    st.subheader("Rating Trends Over Time")
    fig, ax = plt.subplots(figsize=(12, 6))
    for condition in filtered_data['condition'].unique():
        condition_data = filtered_data[filtered_data['condition'] == condition]
        trend = condition_data.groupby(condition_data['date'].dt.to_period("M"))['rating'].mean()
        ax.plot(trend.index.astype(str), trend.astype(float), label=condition, marker='o')
    ax.set_title('Average Rating Trends Over Time')
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Sentiment Insights Tab
with tab3:
    st.header("Sentiment Insights")
    
    if not filtered_data.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sentiment Distribution")
            sentiment_counts = filtered_data['sentiment'].value_counts()
            fig, ax = plt.subplots(figsize=(8, 8))
            colors = ['#4CAF50', '#8BC34A', '#FFC107', '#FF9800', '#F44336']
            ax.pie(sentiment_counts, labels=sentiment_counts.index, 
                  autopct='%1.1f%%', colors=colors, startangle=90)
            st.pyplot(fig)
        
        with col2:
            st.subheader("Sentiment by Condition")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(data=filtered_data, x='condition', hue='sentiment', 
                         palette="Set2", ax=ax)
            ax.set_title('Sentiment Distribution by Condition')
            ax.legend(title='Sentiment', bbox_to_anchor=(1.05, 1))
            st.pyplot(fig)
        
        st.subheader("Sentiment vs Rating Analysis")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=filtered_data, x='sentiment', y='rating', 
                   order=['Strongly Negative', 'Negative', 'Neutral', 
                         'Positive', 'Strongly Positive'],
                   palette="RdYlGn", ax=ax)
        ax.set_title('Rating Distribution by Sentiment')
        st.pyplot(fig)
        
        st.subheader("Review Examples by Sentiment")
        sentiment_choice = st.selectbox("Select sentiment:", 
                                      ['Strongly Positive', 'Positive', 'Neutral', 
                                       'Negative', 'Strongly Negative'])
        sentiment_reviews = filtered_data[filtered_data['sentiment'] == sentiment_choice]['review']
        if not sentiment_reviews.empty:
            for i, review in enumerate(sentiment_reviews.sample(min(3, len(sentiment_reviews))), 1):
                st.markdown(f"""
                <div style="background-color:#f0f2f6; padding:10px; border-radius:5px; margin-bottom:10px;">
                <b>Example {i}:</b> {review}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning(f"No {sentiment_choice} reviews found")

# Condition Comparison Tab
with tab4:
    st.header("Condition Comparison")
    
    if not filtered_data.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Average Rating by Condition")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=filtered_data, x='condition', y='rating', 
                       palette="viridis", ci=None, ax=ax)
            ax.set_title('Average Rating by Condition')
            for p in ax.patches:
                ax.annotate(f"{p.get_height():.1f}", 
                           (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='center', xytext=(0, 10), 
                           textcoords='offset points')
            st.pyplot(fig)
        
        with col2:
            st.subheader("Usefulness by Condition")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=filtered_data, x='condition', y='usefulCount', 
                       palette="magma", ci=None, ax=ax)
            ax.set_title('Average Useful Count by Condition')
            for p in ax.patches:
                ax.annotate(f"{p.get_height():.1f}", 
                           (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='center', xytext=(0, 10), 
                           textcoords='offset points')
            st.pyplot(fig)
        
        st.subheader("Top Drugs by Condition")
        condition_choice = st.selectbox("Select condition:", 
                                      filtered_data['condition'].unique())
        condition_data = filtered_data[filtered_data['condition'] == condition_choice]
        top_drugs = condition_data.groupby('drugName')['rating'].mean().nlargest(5)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(y=top_drugs.index, x=top_drugs.values, palette="coolwarm", ax=ax)
        ax.set_title(f'Top 5 Drugs for {condition_choice}')
        ax.set_xlabel('Average Rating')
        st.pyplot(fig)

# Review Analysis Tab
with tab5:
    st.header("Review Analysis")
    
    if not filtered_data.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Word Cloud by Sentiment")
            sentiment_choice = st.selectbox("Select sentiment:", 
                                          ['All'] + filtered_data['sentiment'].unique().tolist())
            
            if sentiment_choice == 'All':
                text = " ".join(review for review in filtered_data['review'])
                title = "All Reviews"
            else:
                text = " ".join(review for review in 
                               filtered_data[filtered_data['sentiment'] == sentiment_choice]['review'])
                title = f"{sentiment_choice} Reviews"
            
            wordcloud = WordCloud(width=800, height=400, background_color='white',
                                stopwords=STOPWORDS, max_words=100, colormap='viridis').generate(text)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.set_title(title, pad=20)
            ax.axis('off')
            st.pyplot(fig)
        
        with col2:
            st.subheader("Word Cloud by Condition")
            condition_choice = st.selectbox("Select condition:", 
                                          ['All'] + filtered_data['condition'].unique().tolist())
            
            if condition_choice == 'All':
                text = " ".join(review for review in filtered_data['review'])
                title = "All Conditions"
            else:
                text = " ".join(review for review in 
                               filtered_data[filtered_data['condition'] == condition_choice]['review'])
                title = f"{condition_choice} Reviews"
            
            wordcloud = WordCloud(width=800, height=400, background_color='white',
                                stopwords=STOPWORDS, max_words=100, colormap='plasma').generate(text)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.set_title(title, pad=20)
            ax.axis('off')
            st.pyplot(fig)

# Data Explorer Tab
with tab6:
    st.header("Data Explorer")
    
    st.subheader("Filtered Data")
    st.dataframe(filtered_data)
    
    st.subheader("Export Data")
    if st.button("Download as CSV"):
        csv = filtered_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="drug_reviews.csv",
            mime="text/csv"
        )

# Drug Recommendation Tab
with tab7:
    st.header("Personalized Drug Recommendation")
    
    with st.expander("💡 Symptom Examples"):
        st.markdown("""
        - **Depression:** sadness, fatigue, hopelessness
        - **High BP:** headache, dizziness, nosebleeds
        - **Diabetes:** thirst, frequent urination, blurry vision
        """)
    
    symptoms_input = st.text_area(
        "Describe your symptoms:", 
        "headache dizziness",
        help="Enter symptoms separated by commas or spaces"
    )
    
    if st.button("Get Recommendations", type="primary"):
        if symptoms_input.strip():
            with st.spinner("Analyzing symptoms and finding best options..."):
                top_drugs, condition, drug_sentiments, disease_drugs = recommend_drugs(
                    symptoms_input, data, vectorizer, disease_model, disease_encoder)
                
            if top_drugs is not None:
                st.success(f"Most likely condition: **{condition}**")
                
                st.subheader("Recommended Medications")
                for i, (drug, row) in enumerate(top_drugs.iterrows(), 1):
                    with st.expander(f"{i}. {drug} (Score: {row['score']:.1f})"):
                        cols = st.columns(2)
                        with cols[0]:
                            st.metric("Avg Rating", f"{row['rating']:.1f}/10")
                            st.metric("Useful Reviews", row['review'])
                        with cols[1]:
                            st.metric("Usefulness Score", f"{row['usefulCount']:.1f}")
                        
                        if drug in drug_sentiments:
                            sentiments = drug_sentiments[drug]
                            fig, ax = plt.subplots()
                            ax.pie(
                                sentiments.values(),
                                labels=sentiments.keys(),
                                autopct='%1.1f%%',
                                colors=['#4CAF50', '#8BC34A', '#FFC107', '#FF9800', '#F44336']
                            )
                            st.pyplot(fig)
                        
                        reviews = disease_drugs[disease_drugs['drugName'] == drug]['review']
                        if not reviews.empty:
                            st.write("**Patient Experiences:**")
                            for review in reviews.sample(min(3, len(reviews))):
                                highlighted = re.sub(
                                    r'(side effect|dizziness|nausea|headache|fatigue)',
                                    r'<span style="color:red;font-weight:bold">\1</span>',
                                    review,
                                    flags=re.IGNORECASE
                                )
                                st.markdown(f"- {highlighted}", unsafe_allow_html=True)
            else:
                st.warning("No suitable medications found for the predicted condition")
        else:
            st.warning("Please enter symptoms to get recommendations")

# Footer
st.sidebar.markdown("""
### Model Information
- **Algorithm:** Random Forest Classifier
- **Features:** Symptom descriptions (TF-IDF vectorized)
- **Target:** Medical condition
""")

st.markdown("---")
st.markdown("""
**Clinical Decision Support System**  
*This tool provides informational recommendations only and should not replace professional medical advice.*
""")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from wordcloud import WordCloud, STOPWORDS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import spacy

# Load English NLP model
nlp = spacy.load("en_core_web_sm")

# Page configuration
st.set_page_config(
    page_title="Drug Review Analysis",
    page_icon="💊",
    layout="wide"
)

# Title and description
st.title("💊 Drug Review Analysis Dashboard")
st.markdown("""
Analyze patient reviews for drugs treating Depression, High Blood Pressure, and Type 2 Diabetes.
""")

# Sidebar for file upload and parameters
with st.sidebar:
    st.header("Upload Data")
    uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx"])
    
    st.header("Analysis Parameters")
    max_features = st.slider("Max TF-IDF Features", 500, 3000, 2000, 100)
    test_size = st.slider("Test Size Ratio", 0.1, 0.5, 0.2, 0.05)

# Load data function
@st.cache_data
def load_data(file):
    data = pd.read_excel(file)
    # Preprocessing steps from original code
    data.dropna(subset=['review', 'rating', 'drugName'], inplace=True)
    target_conditions = ['Depression', 'High Blood Pressure', 'Diabetes, Type 2']
    data = data[data['condition'].isin(target_conditions)]
    return data

# Text cleaning function
def clean_and_lemmatize(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and len(token.lemma_) > 2]
    return " ".join(tokens)

# Sentiment analysis function
def get_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(text)
    return score['compound']

# Main app logic
if uploaded_file is not None:
    # Load and process data
    data = load_data(uploaded_file)
    
    # Data overview section
    st.header("📊 Data Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("First 5 Rows")
        st.dataframe(data.head())
    
    with col2:
        st.subheader("Dataset Info")
        st.text(f"Shape: {data.shape}")
        st.text(f"Conditions: {data['condition'].nunique()}")
        st.text(f"Unique Drugs: {data['drugName'].nunique()}")
    
    # Visualizations section
    st.header("📈 Visualizations")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Rating Distribution", 
        "Top Drugs", 
        "Condition Analysis",
        "Review Analysis"
    ])
    
    with tab1:
        st.subheader("Rating Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data['rating'], bins=20, kde=True, color='skyblue', ax=ax)
        st.pyplot(fig)
    
    with tab2:
        st.subheader("Top 20 Drugs Reviewed")
        fig, ax = plt.subplots(figsize=(12, 6))
        data['drugName'].value_counts().nlargest(20).plot(
            kind='bar', color='coral', ax=ax
        )
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab3:
        st.subheader("Rating by Condition")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(
            x='condition', y='rating', 
            data=data, palette='Set2', ax=ax
        )
        plt.xticks(rotation=30)
        plt.tight_layout()
        st.pyplot(fig)
        
        st.subheader("Condition Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(
            data=data, x='condition', 
            palette='Set2', ax=ax
        )
        st.pyplot(fig)
    
    with tab4:
        st.subheader("Review Length Distribution")
        data['review_length'] = data['review'].str.len()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(
            data['review_length'], bins=50, 
            kde=True, color='purple', ax=ax
        )
        st.pyplot(fig)
        
        st.subheader("Word Clouds by Condition")
        condition = st.selectbox(
            "Select Condition", 
            data['condition'].unique()
        )
        text = ' '.join(data[data['condition'] == condition]['review'])
        wordcloud = WordCloud(
            width=800, height=400, 
            background_color='white',
            stopwords=STOPWORDS, 
            colormap='plasma'
        ).generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    
    # NLP and Modeling section
    st.header("🤖 NLP & Machine Learning")
    
    # Text preprocessing
    st.subheader("Text Preprocessing")
    if st.button("Clean and Lemmatize Reviews"):
        data['clean_lemma_review'] = data['review'].apply(clean_and_lemmatize)
        st.success("Text preprocessing completed!")
        
        # Show sample
        st.subheader("Sample Processed Reviews")
        sample = data[['review', 'clean_lemma_review']].sample(5)
        st.dataframe(sample)
    
    # Sentiment analysis
    if 'clean_lemma_review' in data.columns:
        st.subheader("Sentiment Analysis")
        data['sentiment_score'] = data['clean_lemma_review'].apply(get_sentiment)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sentiment Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(
                data['sentiment_score'], bins=20, 
                kde=True, color='green', ax=ax
            )
            st.pyplot(fig)
        
        with col2:
            st.subheader("Sentiment vs Rating")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(
                x='rating', y='sentiment_score', 
                data=data, alpha=0.6, ax=ax
            )
            st.pyplot(fig)
    
        # Model training
        st.subheader("Model Training")
        
        # Prepare features
        tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1,2))
        X_tfidf = tfidf.fit_transform(data['clean_lemma_review'])
        
        # Scale sentiment scores
        scaler = MinMaxScaler()
        scaled_sentiment = scaler.fit_transform(data[['sentiment_score']])
        X_final = hstack([X_tfidf, scaled_sentiment])
        y = data['condition']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_final, y, test_size=test_size, stratify=y, random_state=42
        )
        
        # Model selection
        model_option = st.selectbox(
            "Select Model",
            ["Logistic Regression", "Random Forest", "Naive Bayes"]
        )
        
        if st.button("Train Model"):
            if model_option == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)
            elif model_option == "Random Forest":
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:  # Naive Bayes
                model = MultinomialNB()
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Display results
            st.subheader("Model Performance")
            st.text(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
            
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))
            
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap='Blues',
                xticklabels=model.classes_, yticklabels=model.classes_, ax=ax
            )
            st.pyplot(fig)
else:
    st.info("Please upload an Excel file to begin analysis.")
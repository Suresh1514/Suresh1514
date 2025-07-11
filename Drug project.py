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
            'Depression': 'sadness, fatigue, insomnia, loss of interest',
            'High Blood Pressure': 'headache, dizziness, blurred vision, shortness of breath',
            'Diabetes, Type 2': 'thirst, frequent urination, fatigue, blurry vision'
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

# Drug recommendation function
def recommend_drugs(symptoms, data, vectorizer, disease_model, disease_encoder):
    # Preprocess input symptoms
    processed_symptoms = preprocess_text(symptoms)
    
    # Predict disease
    symptoms_vec = vectorizer.transform([processed_symptoms])
    disease_encoded = disease_model.predict(symptoms_vec)
    disease = disease_encoder.inverse_transform(disease_encoded)[0]
    
    # Filter drugs for the predicted disease
    disease_drugs = data[data['condition'] == disease]
    
    if disease_drugs.empty:
        return None, disease, None, None
    
    # Get top drugs by average rating
    top_drugs = disease_drugs.groupby('drugName').agg({
        'rating': 'mean',
        'usefulCount': 'mean',
        'review': 'count'
    }).sort_values(by=['rating', 'usefulCount'], ascending=False).head(5)
    
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

# Sidebar filters (from original file)
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

# Drug Recommendation tab
with tab7:
    st.header("Drug Recommendation Based on Symptoms")
    
    # User input for symptoms
    symptoms_input = st.text_area("Enter your symptoms (comma separated):", 
                                "fatigue, sadness, insomnia")
    
    if st.button("Get Recommendation"):
        if symptoms_input.strip():
            # Get recommendations using the full dataset (not filtered)
            top_drugs, predicted_disease, drug_sentiments, disease_drugs = recommend_drugs(
                symptoms_input, data, vectorizer, disease_model, disease_encoder)
            
            if top_drugs is not None:
                st.success(f"Predicted Condition: {predicted_disease}")
                
                # Display top recommended drugs
                st.subheader("Top Recommended Drugs")
                for i, (drug, row) in enumerate(top_drugs.iterrows(), 1):
                    with st.expander(f"{i}. {drug} (Avg Rating: {row['rating']:.1f}, Useful Count: {row['usefulCount']:.1f})"):
                        st.write(f"**Number of Reviews:** {row['review']}")
                        
                        # Display sentiment distribution
                        st.write("**Sentiment Distribution:**")
                        sentiments = drug_sentiments.get(drug, {})
                        if sentiments:
                            sentiment_df = pd.DataFrame.from_dict(sentiments, orient='index', columns=['Percentage'])
                            sentiment_df['Percentage'] = (sentiment_df['Percentage'] * 100).round(1)
                            st.dataframe(sentiment_df.style.format({'Percentage': '{:.1f}%'}))
                        
                        # Display sample reviews - filter out any empty reviews
                        drug_reviews = disease_drugs[(disease_drugs['drugName'] == drug) & 
                                                    (disease_drugs['review'].notna())]['review']
                        if not drug_reviews.empty:
                            sample_reviews = drug_reviews.sample(min(3, len(drug_reviews)))
                            st.write("**Sample Reviews:**")
                            for review in sample_reviews:
                                st.write(f"- {review}")
                        else:
                            st.write("No reviews available for this drug.")
            else:
                st.warning("No drugs found for the predicted condition.")
        else:
            st.warning("Please enter symptoms to get recommendations.")
# Overview tab (using filtered_data)
with tab1:
    st.header("Data Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Reviews", len(filtered_data))
        st.metric("Unique Drugs", filtered_data['drugName'].nunique())
    
    with col2:
        avg_rating = filtered_data['rating'].mean()
        st.metric("Average Rating", f"{avg_rating:.1f}")
        st.metric("Most Reviewed Condition", filtered_data['condition'].value_counts().idxmax())
    
    with col3:
        most_common_sentiment = filtered_data['sentiment'].value_counts().idxmax()
        st.metric("Most Common Sentiment", most_common_sentiment)
        st.metric("Average Useful Count", f"{filtered_data['usefulCount'].mean():.1f}")
    
    st.subheader("Top 10 Drugs by Review Count")
    top_drugs = filtered_data['drugName'].value_counts().nlargest(10)
    fig, ax = plot.subplots(figsize=(10, 6))
    sns.barplot(y=top_drugs.index, x=top_drugs.values, palette="viridis", ax=ax)
    ax.set_title('Top 10 Drugs by Review Count')
    ax.set_xlabel('Number of Reviews')
    ax.set_ylabel('Drug Name')
    st.pyplot(fig)
    
    st.subheader("Sample Data")
    st.dataframe(filtered_data.head())

# Rating Analysis tab (using filtered_data)
with tab2:
    st.header("Rating Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Rating Distribution by Condition")
        fig, ax = plot.subplots(figsize=(10, 6))
        sns.boxplot(data=filtered_data, x='condition', y='rating', palette="Set2", ax=ax)
        ax.set_title('Rating Distribution by Medical Condition')
        ax.set_xlabel('Medical Condition')
        ax.set_ylabel('Rating')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Rating Distribution")
        fig, ax = plot.subplots(figsize=(10, 6))
        sns.histplot(data=filtered_data, x='rating', hue='condition', 
                     multiple='stack', bins=10, palette="Set2", ax=ax)
        ax.set_title('Stacked Rating Distribution by Condition')
        ax.set_xlabel('Rating')
        ax.set_ylabel('Count')
        st.pyplot(fig)
    
    st.subheader("Rating Trends Over Time")
    fig, ax = plot.subplots(figsize=(12, 6))
    for condition in filtered_data['condition'].unique():
        condition_data = filtered_data[filtered_data['condition'] == condition]
        trend = condition_data.groupby(condition_data['date'].dt.to_period("M"))['rating'].mean()
        ax.plot(trend.index.astype(str), trend.astype(float), label=condition, marker='o')
    
    ax.set_title('Average Rating Trends Over Time by Condition')
    ax.set_xlabel('Month')
    ax.set_ylabel('Average Rating')
    ax.legend()
    plot.xticks(rotation=45)
    st.pyplot(fig)

# Sentiment Analysis tab (using filtered_data)
with tab3:
    st.header("Sentiment Analysis")
    
    if not filtered_data.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sentiment Distribution")
            sentiment_counts = filtered_data['sentiment'].value_counts()
            fig, ax = plot.subplots(figsize=(8, 8))
            colors = ['#4CAF50', '#8BC34A', '#FFC107', '#FF9800', '#F44336']
            ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%',
                  colors=colors, startangle=90)
            ax.set_title('Sentiment Distribution')
            st.pyplot(fig)
        
        with col2:
            st.subheader("Sentiment by Condition")
            fig, ax = plot.subplots(figsize=(10, 6))
            sns.countplot(data=filtered_data, x='condition', hue='sentiment', 
                          palette="Set2", ax=ax)
            ax.set_title('Sentiment Distribution by Condition')
            ax.set_xlabel('Medical Condition')
            ax.set_ylabel('Count')
            ax.legend(title='Sentiment')
            st.pyplot(fig)
        
        st.subheader("Sentiment vs Rating Analysis")
        fig, ax = plot.subplots(figsize=(10, 6))
        sns.boxplot(data=filtered_data, x='sentiment', y='rating', 
                   order=['Negative', 'Slightly Negative', 'Neutral', 
                         'Slightly Positive', 'Positive'],
                   palette="RdYlGn", ax=ax)
        ax.set_title('Rating Distribution by Sentiment')
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Rating')
        st.pyplot(fig)
        
        st.subheader("Review Examples by Sentiment")
        sentiment_choice = st.selectbox("Select sentiment to view examples", 
                                      ['Positive', 'Slightly Positive', 'Neutral', 
                                       'Slightly Negative', 'Negative'])
        
        sentiment_reviews = filtered_data[filtered_data['sentiment'] == sentiment_choice]['review']
        
        if not sentiment_reviews.empty:
            examples = sentiment_reviews.sample(min(5, len(sentiment_reviews)))
            for i, example in enumerate(examples, 1):
                st.markdown(f"""
                <div style="background-color:#c2ee96; padding:10px; border-radius:5px; margin-bottom:10px;">
                <b>Example {i}:</b> {example}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning(f"No {sentiment_choice} reviews found for selected filters")
    else:
        st.warning("No data available for selected filters")

# Condition Comparison tab (using filtered_data)
with tab4:
    st.header("Condition Comparison")
    
    if not filtered_data.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Average Rating by Condition")
            fig, ax = plot.subplots(figsize=(10, 6))
            sns.barplot(data=filtered_data, x='condition', y='rating', 
                       palette="viridis", ci=None, ax=ax)
            ax.set_title('Average Rating by Medical Condition')
            ax.set_xlabel('Medical Condition')
            ax.set_ylabel('Average Rating')
            for p in ax.patches:
                ax.annotate(f"{p.get_height():.1f}", 
                           (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='center', xytext=(0, 10), 
                           textcoords='offset points')
            st.pyplot(fig)
        
        with col2:
            st.subheader("Usefulness by Condition")
            fig, ax = plot.subplots(figsize=(10, 6))
            sns.barplot(data=filtered_data, x='condition', y='usefulCount', 
                        palette="magma", ci=None, ax=ax)
            ax.set_title('Average Useful Count by Medical Condition')
            ax.set_xlabel('Medical Condition')
            ax.set_ylabel('Average Useful Count')
            for p in ax.patches:
                ax.annotate(f"{p.get_height():.1f}", 
                           (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='center', xytext=(0, 10), 
                           textcoords='offset points')
            st.pyplot(fig)
        
        st.subheader("Drug Effectiveness by Condition")
        condition_choice = st.selectbox("Select condition to view top drugs", 
                                      filtered_data['condition'].unique())
        
        condition_data = filtered_data[filtered_data['condition'] == condition_choice]
        top_drugs = condition_data.groupby('drugName')['rating'].mean().nlargest(5)
        
        fig, ax = plot.subplots(figsize=(10, 6))
        sns.barplot(y=top_drugs.index, x=top_drugs.values, palette="coolwarm", ax=ax)
        ax.set_title(f'Top 5 Drugs for {condition_choice} by Average Rating')
        ax.set_xlabel('Average Rating')
        ax.set_ylabel('Drug Name')
        st.pyplot(fig)

# Word Cloud tab (using filtered_data)
with tab5:
    st.header("Review Word Cloud")
    
    if not filtered_data.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            sentiment_choice = st.selectbox("Select sentiment for word cloud", 
                                          ['All', 'Positive', 'Slightly Positive', 
                                           'Neutral', 'Slightly Negative', 'Negative'])
            
            if sentiment_choice == 'All':
                text = " ".join(review for review in filtered_data['review'])
                title = "Word Cloud for All Reviews"
            else:
                text = " ".join(review for review in 
                              filtered_data[filtered_data['sentiment'] == sentiment_choice]['review'])
                title = f"Word Cloud for {sentiment_choice} Reviews"
            
            wordcloud = WordCloud(width=800, height=400, background_color='Green', 
                                stopwords=STOPWORDS, max_words=100, colormap='viridis').generate(text)
            
            fig, ax = plot.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.set_title(title, pad=20)
            ax.axis('off')
            st.pyplot(fig)
        
        with col2:
            condition_choice = st.selectbox("Select condition for word cloud", 
                                          ['All'] + filtered_data['condition'].unique().tolist())
            
            if condition_choice == 'All':
                text = " ".join(review for review in filtered_data['review'])
                title = "Word Cloud for All Conditions"
            else:
                text = " ".join(review for review in 
                              filtered_data[filtered_data['condition'] == condition_choice]['review'])
                title = f"Word Cloud for {condition_choice} Reviews"
            
            wordcloud = WordCloud(width=800, height=400, background_color='Green', 
                                stopwords=STOPWORDS, max_words=100, colormap='plasma').generate(text)
            
            fig, ax = plot.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.set_title(title, pad=20)
            ax.axis('off')
            st.pyplot(fig)

# Data Explorer tab (using filtered_data)
with tab6:
    st.header("Data Explorer")
    
    st.subheader("Filtered Data")
    st.dataframe(filtered_data)
    
    st.subheader("Export Data")
    if st.button("Download Filtered Data as CSV"):
        csv = filtered_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="filtered_drug_reviews.csv",
            mime="text/csv"
        )

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

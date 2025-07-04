import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import re
from textblob import TextBlob  # For sentiment analysis

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
        # Use pd.read_excel() for Excel files
        data = pd.read_excel("drugsCom_raw (2) (1).xlsx") 
    except Exception as e:
        st.warning(f"Using sample data as the real dataset wasn't found. Error: {e}")
        data = pd.DataFrame({
            'drugName': ['Prozac', 'Lisinopril', 'Metformin', 'Zoloft', 'Amlodipine'],
            'condition': ['Depression', 'High Blood Pressure', 'Diabetes, Type 2', 
                        'Depression', 'High Blood Pressure'],
            'review': ["This medication changed my life!", 
                      "Helped lower my blood pressure but caused dizziness",
                      "Effective for sugar control but upset my stomach",
                      "Didn't work for me, made me more anxious",
                      "Works well with minimal side effects"],
            'rating': [9, 7, 6, 3, 8],
            'date': pd.to_datetime(['2020-01-15', '2021-03-22', '2022-05-10', 
                                  '2020-11-30', '2021-07-14']),
            'usefulCount': [45, 32, 28, 19, 37]
        })
    
    # Filter for target conditions
    target_conditions = ['Depression', 'High Blood Pressure', 'Diabetes, Type 2']
    data = data[data['condition'].isin(target_conditions)]
    
    return data


# Sentiment analysis function
def analyze_sentiment(text):
    analysis = TextBlob(str(text))
    if analysis.sentiment.polarity > 0.1:
        return 'Positive'
    elif analysis.sentiment.polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

# Load data
data = load_data()

# Title and description
st.title("💊 Drug Reviews Analysis Dashboard")
st.markdown("""
This dashboard analyzes patient reviews of medications for Depression, High Blood Pressure, and Diabetes (Type 2).
Explore the data through interactive visualizations and sentiment analysis.
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

# Main content
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Rating Analysis", "Sentiment Analysis", "Word Cloud", "Data Explorer"])

with tab1:
    st.header("Data Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Summary")
        st.write(f"Total reviews: {len(data)}")
        st.write(f"Filtered reviews: {len(filtered_data)}")
        st.write(f"Unique drugs: {filtered_data['drugName'].nunique()}")
        st.write(f"Average rating: {filtered_data['rating'].mean():.1f}")
    
    with col2:
        st.subheader("Top Drugs by Review Count")
        top_drugs = filtered_data['drugName'].value_counts().nlargest(10)
        st.bar_chart(top_drugs)
    
    st.subheader("Sample Data")
    st.dataframe(filtered_data.head())

with tab2:
    st.header("Rating Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Rating Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(filtered_data['rating'], bins=20, kde=True, color='skyblue', ax=ax)
        ax.set_title('Distribution of Ratings')
        ax.set_xlabel('Rating')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Rating by Condition")
        rating_by_condition = filtered_data.groupby('condition')['rating'].mean().sort_values(ascending=False)
        st.bar_chart(rating_by_condition)
        
    st.subheader("Rating Trends Over Time")
    rating_trend = filtered_data.groupby(filtered_data['date'].dt.to_period("M"))['rating'].mean()
    st.line_chart(rating_trend.astype(float))

with tab3:
    st.header("Sentiment Analysis")
    
    if not filtered_data.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sentiment Distribution")
            sentiment_counts = filtered_data['sentiment'].value_counts()
            fig, ax = plt.subplots()
            ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%')
            st.pyplot(fig)
        
        with col2:
            st.subheader("Sentiment by Condition")
            sentiment_by_condition = filtered_data.groupby(['condition', 'sentiment']).size().unstack()
            st.bar_chart(sentiment_by_condition)
        
        st.subheader("Review Examples by Sentiment")
        sentiment_choice = st.selectbox("Select sentiment to view examples", ['Positive', 'Neutral', 'Negative'])
        
        # Get reviews for selected sentiment
        sentiment_reviews = filtered_data[filtered_data['sentiment'] == sentiment_choice]['review']
        
        # Determine safe sample size
        sample_size = min(5, len(sentiment_reviews))
        
        # Get examples if available
        if not sentiment_reviews.empty:
            examples = sentiment_reviews.sample(sample_size)
            for i, example in enumerate(examples, 1):
                st.write(f"{i}. {example}")
        else:
            st.warning(f"No {sentiment_choice} reviews found for selected filters")
    else:
        st.warning("No data available for selected filters")

with tab4:
    st.header("Review Word Cloud")
    
    if not filtered_data.empty:
        sentiment_choice = st.selectbox("Select sentiment for word cloud", ['All', 'Positive', 'Neutral', 'Negative'])
        
        if sentiment_choice == 'All':
            text = " ".join(review for review in filtered_data['review'])
        else:
            text = " ".join(review for review in filtered_data[filtered_data['sentiment'] == sentiment_choice]['review'])
        
        wordcloud = WordCloud(width=800, height=400, background_color='white', 
                            stopwords=STOPWORDS, max_words=100).generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    else:
        st.warning("No data available for selected filters")

with tab5:
    st.header("Data Explorer")
    
    st.subheader("Filtered Data")
    st.dataframe(filtered_data)
    
    st.subheader("Export Data")
    if st.button("Download Filtered Data as CSV"):
        csv = filtered_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="C:\\Users\\sures_jp6cuxd\\Desktop\\Drug test Project\\drugsCom_raw (2) (1).xlsx",
            mime="text/csv"
        )

# Add some space at the bottom
st.markdown("---")
st.markdown("""
**Note:** This application analyzes patient reviews for Depression, High Blood Pressure, and Diabetes (Type 2) medications.
""")

# Add requirements.txt for deployment
st.sidebar.markdown("""
**Deployment Requirements:**
streamlit
pandas
numpy
matplotlib
seaborn
wordcloud
textblob
""")

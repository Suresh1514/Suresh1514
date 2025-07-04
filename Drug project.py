import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import re
from textblob import TextBlob
from datetime import datetime

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

# Load data
data = load_data()

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

# Main content
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview", "Rating Analysis", "Sentiment Analysis", 
    "Condition Comparison", "Word Cloud", "Data Explorer"
])

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
                <div style="background-color:#f0f2f6; padding:10px; border-radius:5px; margin-bottom:10px;">
                <b>Example {i}:</b> {example}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning(f"No {sentiment_choice} reviews found for selected filters")
    else:
        st.warning("No data available for selected filters")

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
            
            wordcloud = WordCloud(width=800, height=400, background_color='white', 
                                stopwords=STOPWORDS, max_words=100, colormap='plasma').generate(text)
            
            fig, ax = plot.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.set_title(title, pad=20)
            ax.axis('off')
            st.pyplot(fig)

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

print("\n Top 20 Most Reviewed Drugs:")
print(data['drugName'].value_counts().head(20))

# Count how many unique drugs exist
print(f"\nTotal unique drugs: {data['drugName'].nunique()}")

# 2. Class distribution of target variable (condition)
print("\n Distribution of Conditions:")
print(data['condition'].value_counts())

#  Plot class imbalance
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
sns.countplot(data=data, x='condition', palette='Set2')
plt.title('Condition Class Distribution')
plt.xlabel('Condition')
plt.ylabel('Count')
plt.show()

# Add some space at the bottom
st.markdown("---")
st.markdown("""
**Note:** This application analyzes patient reviews for Depression, High Blood Pressure, and Diabetes (Type 2) medications.
The sentiment analysis categorizes reviews into five categories for more nuanced understanding.
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
""")

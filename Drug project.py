import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import re
pip install streamlit pandas numpy matplotlib seaborn wordcloud


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
    # Replace with your actual data loading method
    # For demo purposes, I'll create a sample dataframe
    data = {
        'drugName': ['Valsartan', 'Guanfacine', 'Lybrel', 'Ortho Evra', 'Buprenorphine / naloxone'],
        'condition': ['Left Ventricular Dysfunction', 'ADHD', 'Birth Control', 'Birth Control', 'Opiate Dependence'],
        'review': ["It has no side effect", "My son is halfway through", "I used to take another", "This is my first time", "Suboxone has completely turned"],
        'rating': [9, 8, 5, 8, 9],
        'date': ['2012-05-20', '2010-04-27', '2009-12-14', '2015-11-03', '2016-11-27'],
        'usefulCount': [27, 192, 17, 10, 37]
    }
    return pd.DataFrame(data)

# Load data
data = load_data()

# Title and description
st.title("💊 Drug Reviews Analysis Dashboard")
st.markdown("""
This dashboard analyzes patient reviews of medications for various conditions.
Explore the data through interactive visualizations and insights.
""")

# Sidebar filters
st.sidebar.header("Filter Data")
selected_conditions = st.sidebar.multiselect(
    "Select conditions to analyze",
    options=data['condition'].unique(),
    default=['Depression', 'High Blood Pressure', 'Diabetes, Type 2']
)

min_rating, max_rating = st.sidebar.slider(
    "Select rating range",
    min_value=1,
    max_value=10,
    value=(1, 10)
)

# Filter data based on selections
filtered_data = data[
    (data['condition'].isin(selected_conditions)) & 
    (data['rating'] >= min_rating) & 
    (data['rating'] <= max_rating)
]

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Rating Analysis", "Word Cloud", "Data Explorer"])

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
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(filtered_data['rating'], bins=20, kde=True, color='skyblue', ax=ax)
    ax.set_title('Distribution of Ratings')
    ax.set_xlabel('Rating')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)
    
    st.subheader("Rating by Condition")
    rating_by_condition = filtered_data.groupby('condition')['rating'].mean().sort_values(ascending=False)
    st.bar_chart(rating_by_condition)

with tab3:
    st.header("Review Word Cloud")
    
    # Combine all reviews
    text = " ".join(review for review in filtered_data['review'])
    
    # Create and generate word cloud image
    wordcloud = WordCloud(width=800, height=400, background_color='white', 
                          stopwords=STOPWORDS, max_words=100).generate(text)
    
    # Display the generated image
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

with tab4:
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

# Add some space at the bottom
st.markdown("---")
st.markdown("""
**Note:** This is a demo application. For the full analysis, please connect to the actual dataset.
""")

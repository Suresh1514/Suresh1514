import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import re
from textblob import TextBlob
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
from scipy.sparse import hstack

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

def train_models(data):
    """
    Train enhanced machine learning models for drug recommendation and disease classification
    """
    # Prepare drug recommendation data with additional features
    drug_data = data[['condition', 'review', 'drugName', 'rating', 'sentiment', 'usefulCount']].dropna()
    
    # Combine condition and review as features
    drug_data['features'] = drug_data['condition'] + " " + drug_data['review']
    
    # Encode drug names as labels
    drug_le = LabelEncoder()
    drug_data['drug_label'] = drug_le.fit_transform(drug_data['drugName'])
    
    # Encode sentiment as numerical values
    sentiment_map = {
        'Negative': 0,
        'Slightly Negative': 1,
        'Neutral': 2,
        'Slightly Positive': 3,
        'Positive': 4
    }
    drug_data['sentiment_score'] = drug_data['sentiment'].map(sentiment_map)
    
    # TF-IDF Vectorization for drug recommendation
    drug_vectorizer = TfidfVectorizer(max_features=1500, stop_words='english', ngram_range=(1, 2))
    X_text = drug_vectorizer.fit_transform(drug_data['features'])
    
    # Additional numerical features
    X_num = drug_data[['rating', 'sentiment_score', 'usefulCount']].values
    
    # Normalize numerical features
    scaler = MinMaxScaler()
    X_num_scaled = scaler.fit_transform(X_num)
    
    # Combine features
    X_drug = hstack([X_text, X_num_scaled])
    y_drug = drug_data['drug_label']
    
    # Split data
    X_drug_train, X_drug_test, y_drug_train, y_drug_test = train_test_split(
        X_drug, y_drug, test_size=0.2, random_state=42
    )
    
    # Train drug recommendation model
    drug_model = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced')
    drug_model.fit(X_drug_train, y_drug_train)
    
    # Prepare drug metadata for ranking
    drug_metadata = data.groupby('drugName').agg({
        'rating': 'mean',
        'sentiment': lambda x: x.map(sentiment_map).mean(),
        'usefulCount': 'mean',
        'review': 'count'
    }).rename(columns={'review': 'review_count'})
    
    # Normalize metadata for scoring
    drug_metadata_scaled = pd.DataFrame(
        scaler.fit_transform(drug_metadata),
        columns=drug_metadata.columns,
        index=drug_metadata.index
    )
    
    # Create composite score (weights can be adjusted)
    drug_metadata['composite_score'] = (
        0.4 * drug_metadata_scaled['rating'] +
        0.3 * drug_metadata_scaled['sentiment'] +
        0.2 * drug_metadata_scaled['usefulCount'] +
        0.1 * drug_metadata_scaled['review_count']
    )
    
    # Prepare disease classification data
    disease_data = data[['review', 'condition']].dropna()
    
    # Encode conditions as labels
    disease_le = LabelEncoder()
    disease_data['condition_label'] = disease_le.fit_transform(disease_data['condition'])
    
    # TF-IDF Vectorization for disease classification
    disease_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X_disease = disease_vectorizer.fit_transform(disease_data['review'])
    y_disease = disease_data['condition_label']
    
    # Split data
    X_disease_train, X_disease_test, y_disease_train, y_disease_test = train_test_split(
        X_disease, y_disease, test_size=0.2, random_state=42
    )
    
    # Train disease classification model
    disease_model = RandomForestClassifier(n_estimators=100, random_state=42)
    disease_model.fit(X_disease_train, y_disease_train)
    
    return {
        'drug_model': drug_model,
        'drug_vectorizer': drug_vectorizer,
        'drug_le': drug_le,
        'drug_metadata': drug_metadata,
        'scaler': scaler,
        'sentiment_map': sentiment_map,
        'disease_model': disease_model,
        'disease_vectorizer': disease_vectorizer,
        'disease_le': disease_le
    }

# Load data
data = load_data()

# Add sentiment analysis
data['sentiment'] = data['review'].apply(analyze_sentiment)

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

@st.cache_resource
def get_models():
    data = load_data()
    data['sentiment'] = data['review'].apply(analyze_sentiment)
    return train_models(data)

models = get_models()

# Main content
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Overview", "Rating Analysis", "Sentiment Analysis", 
    "Condition Comparison", "Word Cloud", "Data Explorer", "Prediction"
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
    fig, ax = plt.subplots(figsize=(10, 6))
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
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=filtered_data, x='condition', y='rating', palette="Set2", ax=ax)
        ax.set_title('Rating Distribution by Medical Condition')
        ax.set_xlabel('Medical Condition')
        ax.set_ylabel('Rating')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Rating Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=filtered_data, x='rating', hue='condition', 
                     multiple='stack', bins=10, palette="Set2", ax=ax)
        ax.set_title('Stacked Rating Distribution by Condition')
        ax.set_xlabel('Rating')
        ax.set_ylabel('Count')
        st.pyplot(fig)
    
    st.subheader("Rating Trends Over Time")
    fig, ax = plt.subplots(figsize=(12, 6))
    for condition in filtered_data['condition'].unique():
        condition_data = filtered_data[filtered_data['condition'] == condition]
        trend = condition_data.groupby(condition_data['date'].dt.to_period("M"))['rating'].mean()
        ax.plot(trend.index.astype(str), trend.astype(float), label=condition, marker='o')
    
    ax.set_title('Average Rating Trends Over Time by Condition')
    ax.set_xlabel('Month')
    ax.set_ylabel('Average Rating')
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

with tab3:
    st.header("Sentiment Analysis")
    
    if not filtered_data.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sentiment Distribution")
            sentiment_counts = filtered_data['sentiment'].value_counts()
            fig, ax = plt.subplots(figsize=(8, 8))
            colors = ['#4CAF50', '#8BC34A', '#FFC107', '#FF9800', '#F44336']
            ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%',
                  colors=colors, startangle=90)
            ax.set_title('Sentiment Distribution')
            st.pyplot(fig)
        
        with col2:
            st.subheader("Sentiment by Condition")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(data=filtered_data, x='condition', hue='sentiment', 
                          palette="Set2", ax=ax)
            ax.set_title('Sentiment Distribution by Condition')
            ax.set_xlabel('Medical Condition')
            ax.set_ylabel('Count')
            ax.legend(title='Sentiment')
            st.pyplot(fig)
        
        st.subheader("Sentiment vs Rating Analysis")
        fig, ax = plt.subplots(figsize=(10, 6))
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

with tab4:
    st.header("Condition Comparison")
    
    if not filtered_data.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Average Rating by Condition")
            fig, ax = plt.subplots(figsize=(10, 6))
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
            fig, ax = plt.subplots(figsize=(10, 6))
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
        
        fig, ax = plt.subplots(figsize=(10, 6))
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
            
            wordcloud = WordCloud(width=800, height=400, background_color='white', 
                                stopwords=STOPWORDS, max_words=100, colormap='viridis').generate(text)
            
            fig, ax = plt.subplots(figsize=(10, 5))
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
            
            fig, ax = plt.subplots(figsize=(10, 5))
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

with tab7:
    st.header("Enhanced Drug Recommendation System")
    st.markdown("""
    This system recommends drugs based on:
    - Your medical condition and symptoms
    - Average ratings and sentiment analysis
    - Community usefulness scores
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        input_condition = st.selectbox(
            "Select your medical condition",
            options=data['condition'].unique(),
            index=0
        )
        
        min_rating_filter = st.slider(
            "Minimum average rating for drugs",
            min_value=1.0,
            max_value=10.0,
            value=7.0,
            step=0.5
        )
    
    with col2:
        input_review = st.text_area(
            "Describe your symptoms or what you're looking for in a medication",
            placeholder="e.g., I need something for depression that doesn't cause weight gain...",
            height=100
        )
        
        include_sentiment = st.checkbox(
            "Prioritize drugs with more positive reviews",
            value=True
        )
    
    if st.button("Get Enhanced Recommendations"):
        if input_review:
            with st.spinner("Analyzing and generating recommendations..."):
                # Prepare input
                input_text = f"{input_condition} {input_review}"
                input_vec = models['drug_vectorizer'].transform([input_text])
                
                # Add numerical features (using median values as defaults)
                input_num = np.array([[7.0, 3.0, 30.0]])  # Default: rating=7, sentiment=3 (Slightly Positive), usefulCount=30
                input_num_scaled = models['scaler'].transform(input_num)
                
                # Combine features
                X_input = hstack([input_vec, input_num_scaled])
                
                # Get predictions
                drug_probs = models['drug_model'].predict_proba(X_input)[0]
                top_n = 20  # Get more candidates for filtering
                top_indices = drug_probs.argsort()[-top_n:][::-1]
                candidate_drugs = models['drug_le'].inverse_transform(top_indices)
                
                # Filter and rank candidates
                recommendations = []
                for drug in candidate_drugs:
                    if drug in models['drug_metadata'].index:
                        meta = models['drug_metadata'].loc[drug]
                        if meta['rating'] >= min_rating_filter:
                            if include_sentiment:
                                score = meta['composite_score']
                            else:
                                score = 0.6 * meta['rating'] + 0.4 * meta['usefulCount']
                            
                            recommendations.append({
                                'drug': drug,
                                'score': score,
                                'rating': meta['rating'],
                                'sentiment': meta['sentiment'],
                                'usefulCount': meta['usefulCount'],
                                'review_count': meta['review_count']
                            })
                
                # Sort by score
                recommendations.sort(key=lambda x: x['score'], reverse=True)
                
                # Display top 5 recommendations
                st.subheader("Top Recommended Drugs")
                st.markdown("""
                <style>
                .drug-card {
                    border-radius: 10px;
                    padding: 15px;
                    margin-bottom: 15px;
                    background-color: #f8f9fa;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .drug-name {
                    font-size: 1.2em;
                    font-weight: bold;
                    color: #2c3e50;
                }
                .rating {
                    color: #e67e22;
                    font-weight: bold;
                }
                .sentiment {
                    font-style: italic;
                }
                .positive { color: #27ae60; }
                .negative { color: #e74c3c; }
                </style>
                """, unsafe_allow_html=True)
                
                for i, rec in enumerate(recommendations[:5], 1):
                    # Determine sentiment label
                    sentiment_value = rec['sentiment']
                    if sentiment_value >= 3.5:
                        sentiment_label = "Very Positive"
                        sentiment_class = "positive"
                    elif sentiment_value >= 2.5:
                        sentiment_label = "Generally Positive"
                        sentiment_class = "positive"
                    elif sentiment_value >= 1.5:
                        sentiment_label = "Mixed Reviews"
                        sentiment_class = ""
                    else:
                        sentiment_label = "Mostly Negative"
                        sentiment_class = "negative"
                    
                    # Create drug card
                    st.markdown(f"""
                    <div class="drug-card">
                        <div class="drug-name">{i}. {rec['drug']}</div>
                        <div>Average Rating: <span class="rating">{rec['rating']:.1f}/10</span></div>
                        <div>Sentiment: <span class="sentiment {sentiment_class}">{sentiment_label}</span></div>
                        <div>Based on {int(rec['review_count'])} reviews | Useful Count: {rec['usefulCount']:.1f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show top reviews for this drug
                    drug_reviews = data[data['drugName'] == rec['drug']]
                    if not drug_reviews.empty:
                        with st.expander(f"See representative reviews for {rec['drug']}"):
                            # Get one review for each sentiment category
                            for sentiment in ['Positive', 'Slightly Positive', 'Neutral', 'Slightly Negative', 'Negative']:
                                sentiment_reviews = drug_reviews[drug_reviews['sentiment'] == sentiment]
                                if not sentiment_reviews.empty:
                                    review = sentiment_reviews.sample(1).iloc[0]
                                    st.markdown(f"""
                                    <div style="margin: 5px 0; padding: 8px; 
                                                border-left: 3px solid {'#2ecc71' if 'Positive' in sentiment else '#e74c3c' if 'Negative' in sentiment else '#f39c12'};
                                                background-color: #f5f5f5;">
                                        <div><b>{sentiment} Review</b> (Rating: {review['rating']}/10)</div>
                                        <div>"{review['review']}"</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                if not recommendations:
                    st.warning("No drugs match your criteria. Try adjusting your filters.")
        else:
            st.warning("Please describe your symptoms to get recommendations")
    
    # Add a section for drug comparison
    st.markdown("---")
    st.subheader("Compare Multiple Drugs")
    
    selected_drugs = st.multiselect(
        "Select drugs to compare",
        options=data['drugName'].unique(),
        default=['Prozac', 'Zoloft', 'Lexapro'] if 'Depression' in data['condition'].unique() else []
    )
    
    if selected_drugs:
        comparison_data = []
        for drug in selected_drugs:
            if drug in models['drug_metadata'].index:
                meta = models['drug_metadata'].loc[drug]
                comparison_data.append({
                    'Drug': drug,
                    'Avg Rating': meta['rating'],
                    'Sentiment Score': meta['sentiment'],
                    'Useful Count': meta['usefulCount'],
                    'Review Count': meta['review_count']
                })
        
        if comparison_data:
            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(
                df_comparison.style
                .background_gradient(subset=['Avg Rating'], cmap='YlOrRd')
                .background_gradient(subset=['Sentiment Score'], cmap='RdYlGn')
                .format({
                    'Avg Rating': '{:.1f}',
                    'Sentiment Score': '{:.2f}',
                    'Useful Count': '{:.1f}'
                }),
                height=(len(df_comparison) + 1) * 35 + 3
            )
            
            # Visual comparison
            fig, ax = plt.subplots(figsize=(10, 6))
            x = range(len(selected_drugs))
            width = 0.2
            
            # Plot rating
            ax.bar(x, df_comparison['Avg Rating'], width, label='Avg Rating', color='#e67e22')
            
            # Plot sentiment (scaled to 10)
            ax.bar([i + width for i in x], 
                  df_comparison['Sentiment Score'] * 2.5,  # Scale 0-4 to 0-10
                  width, label='Sentiment (scaled)', color='#2ecc71')
            
            # Plot usefulness (scaled)
            max_useful = df_comparison['Useful Count'].max()
            scale_factor = 10 / max_useful if max_useful > 0 else 1
            ax.bar([i + width*2 for i in x], 
                  df_comparison['Useful Count'] * scale_factor,
                  width, label='Usefulness (scaled)', color='#3498db')
            
            ax.set_xticks([i + width for i in x])
            ax.set_xticklabels(selected_drugs)
            ax.set_ylabel('Score (scaled to 10)')
            ax.set_title('Drug Comparison (Metrics Scaled to 10)')
            ax.legend()
            st.pyplot(fig)
    
    # Disease classification section
    st.markdown("---")
    st.subheader("Disease Classification")
    st.write("Enter a drug review to predict the most likely medical condition")
    
    classify_review = st.text_area(
        "Enter a drug review for classification",
        placeholder="e.g., This medication helped control my blood sugar levels effectively...",
        key="classify_review"
    )
    
    if st.button("Classify Condition"):
        if classify_review:
            # Prepare input
            input_vec = models['disease_vectorizer'].transform([classify_review])
            
            # Get prediction
            prediction = models['disease_model'].predict(input_vec)
            condition = models['disease_le'].inverse_transform(prediction)[0]
            
            # Get confidence
            probabilities = models['disease_model'].predict_proba(input_vec)[0]
            confidence = probabilities.max()
            
            # Display result
            st.subheader("Predicted Condition")
            st.success(f"**{condition}** (confidence: {confidence*100:.1f}%)")
            
            # Show similar reviews
            st.subheader("Similar Reviews")
            similar_reviews = data[data['condition'] == condition]['review'].sample(min(5, len(data[data['condition'] == condition])))
            for review in similar_reviews:
                st.markdown(f"<div style='background-color:#f0f0f0; padding:8px; border-radius:4px; margin:4px 0;'>{review}</div>", 
                            unsafe_allow_html=True)
        else:
            st.warning("Please enter a review to classify")

# Add requirements.txt for deployment
st.sidebar.markdown("""
**Deployment requirements.txt:**

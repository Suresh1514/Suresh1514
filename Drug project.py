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
from sklearn.metrics import classification_report, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import pickle
import streamlit as st

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

@st.cache_data
def load_data():
    try:
        # First try as Excel file
        try:
            df = pd.read_excel('drugsCom_raw.xlsx', engine='openpyxl')
        except:
            # If Excel fails, try as CSV
            try:
                df = pd.read_csv('drugsCom_raw.xlsx')  # Try as CSV even with .xlsx extension
            except:
                st.warning("File is neither valid Excel nor CSV. Using sample data.")
                return create_sample_data()
        
        # Check required columns
        required_columns = ['DrugName', 'condition', 'review', 'rating', 'date', 'usefulCount']
        if not all(col in df.columns for col in required_columns):
            st.warning("Dataset missing required columns. Using sample data.")
            return create_sample_data()
            
        return df
    except Exception as e:
        st.warning(f"Error loading dataset: {str(e)}. Using sample data.")
        return create_sample_data()

def create_sample_data():
    # Fallback sample data
    data = {
        'DrugName': ['Prozac', 'Lisinopril', 'Metformin', 'Prozac', 'Metformin'],
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

df = load_data()



# Preprocessing functions
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

# Apply preprocessing
def apply_preprocessing(df):
    df['cleaned_review'] = df['review'].apply(preprocess_text)
    df['cleaned_review'] = df['cleaned_review'].apply(remove_stopwords)
    df['cleaned_review'] = df['cleaned_review'].apply(lemmatize_text)
    return df

df = apply_preprocessing(df)

# Focus only on the three conditions
target_conditions = ['Depression', 'High Blood Pressure', 'Diabetes, Type 2']
df = df[df['condition'].isin(target_conditions)]

# Prepare data for model training
X = df['cleaned_review']
y = df['condition']

# Vectorize text data
vectorizer = TfidfVectorizer(max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000, multi_class='multinomial')
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Function to recommend drugs based on condition
def recommend_drugs(condition):
    condition_df = df[df['condition'] == condition]
    if condition_df.empty:
        return "No drugs found for this condition"
    
    # Get top 3 drugs by average rating (with at least 5 usefulCount)
    recommended_drugs = condition_df.groupby('DrugName').agg({
        'rating': 'mean',
        'usefulCount': 'sum'
    }).reset_index()
    
    recommended_drugs = recommended_drugs[recommended_drugs['usefulCount'] >= 5]
    recommended_drugs = recommended_drugs.sort_values(by=['rating', 'usefulCount'], ascending=[False, False])
    
    if len(recommended_drugs) > 3:
        recommended_drugs = recommended_drugs.head(3)
    
    return recommended_drugs[['DrugName', 'rating']]

# Save model and vectorizer for deployment
pickle.dump(model, open('drug_classifier_model.pkl', 'wb'))
pickle.dump(vectorizer, open('tfidf_vectorizer.pkl', 'wb'))

# Streamlit App
def main():
    st.title("Drug Recommendation System")
    st.write("This system analyzes patient reviews to classify conditions and recommend appropriate drugs.")
    
    # User input
    user_review = st.text_area("Enter your medical review or symptoms:", height=150)
    
    if st.button("Analyze and Recommend"):
        if user_review:
            # Preprocess the input
            cleaned_review = preprocess_text(user_review)
            cleaned_review = remove_stopwords(cleaned_review)
            cleaned_review = lemmatize_text(cleaned_review)
            
            # Vectorize the input
            review_vectorized = vectorizer.transform([cleaned_review])
            
            # Predict condition
            predicted_condition = model.predict(review_vectorized)[0]
            
            # Check if condition is one of our target conditions
            if predicted_condition in target_conditions:
                st.subheader(f"Predicted Condition: {predicted_condition}")
                
                # Get drug recommendations
                recommendations = recommend_drugs(predicted_condition)
                
                if isinstance(recommendations, str):
                    st.write(recommendations)
                else:
                    st.subheader("Recommended Drugs:")
                    for idx, row in recommendations.iterrows():
                        st.write(f"- {row['DrugName']} (Average Rating: {row['rating']:.1f}/10)")
            else:
                st.write("The review doesn't match our target conditions (Depression, High Blood Pressure, Diabetes Type 2).")
        else:
            st.warning("Please enter a medical review or symptoms.")

if __name__ == '__main__':
    main()

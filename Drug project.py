import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
import joblib
import streamlit as st

# Sample data (replace with your actual dataset loading)
def load_data():
    # This should be replaced with your actual data loading code
    data = pd.DataFrame({
        'review': [
            "This helped with my depression and anxiety",
            "Effective for lowering blood pressure",
            "Controls my blood sugar levels well",
            "Made me feel more anxious and depressed",
            "Reduced my blood pressure but caused dizziness"
        ],
        'drugName': ['Prozac', 'Lisinopril', 'Metformin', 'Zoloft', 'Amlodipine'],
        'condition': ['Depression', 'High Blood Pressure', 'Diabetes', 'Depression', 'High Blood Pressure'],
        'rating': [9, 7, 8, 4, 6]
    })
    return data

# Train or load the recommendation model
def get_recommendation_model(data):
    model_path = "drug_recommendation_model.joblib"
    
    try:
        # Try to load existing model
        model = joblib.load(model_path)
        return model
    except:
        # Prepare data for multi-output prediction
        X = data['review']
        y = data[['drugName', 'condition', 'rating']]
        
        # Convert rating to categories for classification
        y['rating'] = pd.cut(y['rating'], bins=[0, 4, 7, 10], 
                            labels=['Low', 'Medium', 'High'])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create pipeline for multi-output classification
        model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
            ('clf', MultiOutputClassifier(
                RandomForestClassifier(n_estimators=100, random_state=42)
            ))
        ])
        
        # Train model
        model.fit(X_train, y_train)
        
        # Save model
        joblib.dump(model, model_path)
        
        return model

# Predict function
def predict_recommendation(model, review_text):
    if not review_text:
        return None
    
    # Make prediction
    pred = model.predict([review_text])
    
    # Get probabilities for each class (more complex in multi-output)
    # For simplicity, we'll just return the predictions here
    # In a production system, you'd want to get probabilities too
    
    return {
        'drug': pred[0][0],
        'condition': pred[0][1],
        'rating_category': pred[0][2]
    }

# Streamlit interface
def main():
    st.title("Drug Recommendation System")
    st.write("Enter your symptoms or medication experience to get recommendations")
    
    # Load data
    data = load_data()
    
    # Get model
    model = get_recommendation_model(data)
    
    # User input
    review_text = st.text_area("Describe your symptoms or medication experience:")
    
    if st.button("Get Recommendation"):
        if review_text:
            with st.spinner("Analyzing your input..."):
                recommendation = predict_recommendation(model, review_text)
                
                if recommendation:
                    st.subheader("Recommendation")
                    
                    # Display recommendation
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Likely Condition", recommendation['condition'])
                    with col2:
                        st.metric("Recommended Medication", recommendation['drug'])
                    with col3:
                        st.metric("Expected Effectiveness", recommendation['rating_category'])
                    
                    # Show similar reviews from dataset
                    st.subheader("Similar Patient Experiences")
                    similar_reviews = data[
                        (data['condition'] == recommendation['condition']) & 
                        (data['drugName'] == recommendation['drug'])
                    ].head(3)
                    
                    if not similar_reviews.empty:
                        for _, row in similar_reviews.iterrows():
                            st.markdown(f"""
                            <div style="background-color:#f0f2f6; padding:10px; border-radius:5px; margin-bottom:10px;">
                            <b>Rating {row['rating']}/10:</b> {row['review']}
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.warning("No similar experiences found in our database")
                else:
                    st.error("Could not generate recommendation")
        else:
            st.warning("Please describe your symptoms or experience")

if __name__ == "__main__":
    main()

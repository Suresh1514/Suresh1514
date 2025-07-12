import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib
import streamlit as st


# Set page configuration
st.set_page_config(
    page_title="Drug Reviews Analysis",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sample data with specific conditions
def load_data():
    data = pd.DataFrame({
        'review': [
            "This helped with my depression and anxiety",
            "Effective for lowering blood pressure",
            "Controls my blood sugar levels well",
            "Made me feel more anxious and depressed",
            "Reduced my blood pressure but caused dizziness",
            "Helps manage my diabetes effectively",
            "Not working for my depression symptoms",
            "Great for hypertension control",
            "Stabilized my mood and depression",
            "Blood pressure is now under control",
            "My A1C levels improved significantly",
            "Still feeling depressed despite medication"
        ],
        'drugName': ['Prozac', 'Lisinopril', 'Metformin', 'Zoloft', 'Amlodipine',
                    'Empagliflozin', 'Fluoxetine', 'Losartan', 'Sertraline',
                    'Hydrochlorothiazide', 'Glipizide', 'Lexapro'],
        'condition': ['Depression', 'High Blood Pressure', 'Diabetes, Type 2', 
                     'Depression', 'High Blood Pressure', 'Diabetes, Type 2',
                     'Depression', 'High Blood Pressure', 'Depression',
                     'High Blood Pressure', 'Diabetes, Type 2', 'Depression'],
        'rating': [9, 7, 8, 4, 6, 8, 3, 9, 7, 8, 9, 5]
    })
    return data

# Filter conditions to only the three we want
def filter_conditions(data):
    valid_conditions = ['Depression', 'Diabetes, Type 2', 'High Blood Pressure']
    data['condition'] = data['condition'].apply(
        lambda x: x if x in valid_conditions else 'None'
    )
    return data

# Train or load the recommendation model
def get_recommendation_model(data):
    model_path = "condition_prediction_model.joblib"
    
    try:
        model = joblib.load(model_path)
        return model
    except:
        # Prepare data
        data = filter_conditions(data)
        X = data['review']
        y = data['condition']
        
        # Only train if we have enough data
        if len(data) < 10 or y.nunique() < 2:
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create pipeline
        model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        # Train model
        model.fit(X_train, y_train)
        
        # Save model
        joblib.dump(model, model_path)
        
        return model

# Predict function with condition filtering
def predict_condition(model, review_text):
    if not review_text or model is None:
        return None
    
    # Make prediction
    pred = model.predict([review_text])[0]
    
    # Only return our three target conditions, otherwise return None
    valid_conditions = ['Depression', 'Diabetes, Type 2', 'High Blood Pressure']
    return pred if pred in valid_conditions else 'None'

# Streamlit interface
def main():
    st.title("Medical Condition Prediction System")
    st.write("Describe your symptoms to predict possible conditions")
    
    # Load and prepare data
    data = load_data()
    data = filter_conditions(data)
    
    # Get model
    model = get_recommendation_model(data)
    
    # User input
    review_text = st.text_area("Describe your symptoms:")
    
    if st.button("Predict Condition"):
        if review_text:
            with st.spinner("Analyzing your symptoms..."):
                prediction = predict_condition(model, review_text)
                
                if prediction:
                    st.subheader("Predicted Condition")
                    
                    if prediction == 'None':
                        st.warning("The described symptoms don't match our target conditions (Depression, Diabetes Type 2, High Blood Pressure)")
                    else:
                        st.success(f"Predicted condition: {prediction}")
                        
                        # Show relevant medications
                        st.subheader("Common Medications for this Condition")
                        meds = data[data['condition'] == prediction]['drugName'].unique()
                        for med in meds[:5]:  # Show top 5 medications
                            avg_rating = data[
                                (data['condition'] == prediction) & 
                                (data['drugName'] == med)
                            ]['rating'].mean()
                            st.write(f"- {med} (Average rating: {avg_rating:.1f}/10)")
                else:
                    st.error("Could not generate prediction")
        else:
            st.warning("Please describe your symptoms")

if __name__ == "__main__":
    main()

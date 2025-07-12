import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib
import streamlit as st

# Function to load dataset with proper error handling
def load_data():
    try:
        # Load your dataset - replace with your actual file path
        data = pd.read_Excel("drugsCom_raw.xlsx")  # or .xlsx for Excel files
        
        # Clean and standardize condition names
        condition_mapping = {
            'depression': 'Depression',
            'diabetes': 'Diabetes, Type 2',
            'type 2 diabetes': 'Diabetes, Type 2',
            'high blood pressure': 'High Blood Pressure',
            'hypertension': 'High Blood Pressure'
        }
        
        # Convert conditions to standard format
        if 'condition' in data.columns:
            data['condition'] = data['condition'].str.lower().map(condition_mapping).fillna('None')
        else:
            st.error("Dataset is missing 'condition' column")
            return pd.DataFrame()
        
        # Filter for only our target conditions
        valid_conditions = ['Depression', 'Diabetes, Type 2', 'High Blood Pressure']
        data = data[data['condition'].isin(valid_conditions)]
        
        # Verify required columns exist
        required_columns = ['review', 'drugName', 'condition', 'rating']
        for col in required_columns:
            if col not in data.columns:
                st.error(f"Dataset is missing required column: {col}")
                return pd.DataFrame()
                
        return data[required_columns].dropna()
    
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        # Fallback to sample data
        return pd.DataFrame({
            'review': [
                "This helped with my depression and anxiety",
                "Effective for lowering blood pressure",
                "Controls my blood sugar levels well"
            ],
            'drugName': ['Prozac', 'Lisinopril', 'Metformin'],
            'condition': ['Depression', 'High Blood Pressure', 'Diabetes, Type 2'],
            'rating': [9, 7, 8]
        })

# Train or load the prediction model
def get_prediction_model(data):
    model_path = "drug_condition_model.joblib"
    
    try:
        model = joblib.load(model_path)
        return model
    except:
        # Only train if we have sufficient data
        if len(data) < 10 or data['condition'].nunique() < 2:
            return None
            
        X = data['review']
        y = data['condition']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create model pipeline
        model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        # Train model
        model.fit(X_train, y_train)
        
        # Save model
        joblib.dump(model, model_path)
        
        return model

# Prediction function with condition filtering
def predict_condition(model, review_text):
    if not review_text or model is None:
        return None
    
    pred = model.predict([review_text])[0]
    valid_conditions = ['Depression', 'Diabetes, Type 2', 'High Blood Pressure']
    return pred if pred in valid_conditions else 'None'

# Main Streamlit app
def main():
    st.title("Medical Condition Prediction System")
    st.write("Describe your symptoms to predict possible conditions")
    
    # Load data
    data = load_data()
    
    # Train model
    model = get_prediction_model(data)
    
    # User input
    review_text = st.text_area("Describe your symptoms:")
    
    if st.button("Predict Condition"):
        if review_text:
            with st.spinner("Analyzing your symptoms..."):
                prediction = predict_condition(model, review_text)
                
                if prediction:
                    st.subheader("Prediction Results")
                    
                    if prediction == 'None':
                        st.warning("Symptoms don't match target conditions (Depression, Diabetes Type 2, High Blood Pressure)")
                    else:
                        st.success(f"Predicted condition: {prediction}")
                        
                        # Show top medications for predicted condition
                        st.subheader("Common Medications")
                        top_meds = (data[data['condition'] == prediction]
                                   .groupby('drugName')['rating']
                                   .mean()
                                   .sort_values(ascending=False)
                                   .head(5))
                        
                        for med, rating in top_meds.items():
                            st.write(f"- {med} (Avg rating: {rating:.1f}/10)")
                            
                        # Show sample reviews
                        st.subheader("Example Patient Experiences")
                        sample_reviews = data[data['condition'] == prediction].sample(3)
                        for _, row in sample_reviews.iterrows():
                            st.markdown(f"""
                            <div style="background-color:#f0f2f6; padding:10px; border-radius:5px; margin-bottom:10px;">
                            <b>Rating {row['rating']}/10:</b> {row['review']}
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.error("Could not generate prediction")
        else:
            st.warning("Please describe your symptoms")

if __name__ == "__main__":
    main()

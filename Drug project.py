import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib
import streamlit as st
import os

# Function to load dataset with comprehensive error handling
def load_data():
    try:
        # Define possible file locations
        possible_paths = [
            "drugsCom_raw.xlsx",                # Same directory
            "data/drugsCom_raw.xlsx",           # In a data subfolder
            "../drugsCom_raw.xlsx",             # One level up
            "./data/drugsCom_raw.xlsx"          # Current directory data subfolder
        ]
        
        data = None
        file_found = False
        
        # Try all possible file locations
        for path in possible_paths:
            try:
                if os.path.exists(path):
                    st.info(f"Found file at: {path}")
                    data = pd.read_excel(path)
                    file_found = True
                    st.success("Successfully loaded dataset!")
                    break
            except Exception as e:
                st.warning(f"Tried {path} but failed with error: {str(e)}")
                continue
        
        if not file_found:
            st.error("Could not find drugsCom_raw.xlsx in any of these locations:")
            st.code("\n".join(possible_paths))
            st.warning("Using sample data instead")
            
            # Sample data fallback
            sample_data = {
                'review': [
                    "This medication helped with my depression symptoms",
                    "Effective at controlling my blood pressure",
                    "Great for managing my type 2 diabetes",
                    "Didn't help my depression at all",
                    "Lowered my blood pressure but caused dizziness"
                ],
                'drugName': ['Prozac', 'Lisinopril', 'Metformin', 'Zoloft', 'Amlodipine'],
                'condition': ['Depression', 'High Blood Pressure', 'Diabetes, Type 2', 
                            'Depression', 'High Blood Pressure'],
                'rating': [9, 7, 8, 3, 6]
            }
            return pd.DataFrame(sample_data)
        
        # Clean and standardize condition names
        condition_mapping = {
            'depression': 'Depression',
            'diabetes': 'Diabetes, Type 2',
            'type 2 diabetes': 'Diabetes, Type 2',
            'high blood pressure': 'High Blood Pressure',
            'hypertension': 'High Blood Pressure',
            'blood pressure': 'High Blood Pressure',
            'diabetes mellitus, type 2': 'Diabetes, Type 2'
        }
        
        # Convert conditions to standard format
        if 'condition' in data.columns:
            data['condition'] = (data['condition']
                                .astype(str)
                                .str.lower()
                                .str.strip()
                                .map(condition_mapping)
                                )
            # Remove rows with None/mapped conditions
            data = data[data['condition'].isin(['Depression', 'Diabetes, Type 2', 'High Blood Pressure'])]
        else:
            st.error("Your dataset is missing the 'condition' column")
            return pd.DataFrame()
        
        # Verify required columns exist
        required_columns = ['review', 'drugName', 'condition', 'rating']
        missing_cols = [col for col in required_columns if col not in data.columns]
        
        if missing_cols:
            st.error(f"Dataset is missing required columns: {', '.join(missing_cols)}")
            return pd.DataFrame()
        
        return data[required_columns].dropna()
    
    except Exception as e:
        st.error(f"Unexpected error loading data: {str(e)}")
        return pd.DataFrame()

# Train or load the prediction model
def get_prediction_model(data):
    model_path = "drug_condition_model.joblib"
    
    try:
        if os.path.exists(model_path):
            return joblib.load(model_path)
    except:
        st.warning("Could not load existing model, will train new one")
    
    # Only train if we have sufficient data
    if len(data) < 10:
        st.warning("Insufficient data to train model")
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

# Prediction function
def predict_condition(model, review_text):
    if not review_text or model is None:
        return None
    
    pred = model.predict([review_text])[0]
    valid_conditions = ['Depression', 'Diabetes, Type 2', 'High Blood Pressure']
    return pred if pred in valid_conditions else 'None'

# Main Streamlit app
def main():
    st.title("Drug Condition Prediction System")
    st.markdown("Analyze patient reviews to predict medical conditions")
    
    # Load data
    data = load_data()
    
    if data.empty:
        st.error("No valid data available. Please check your dataset.")
        return
    
    # Train model
    model = get_prediction_model(data)
    
    # User interface
    st.sidebar.header("Patient Review Analysis")
    review_text = st.sidebar.text_area("Enter patient review or symptoms:")
    
    if st.sidebar.button("Predict Condition"):
        if review_text:
            with st.spinner("Analyzing review..."):
                prediction = predict_condition(model, review_text)
                
                if prediction:
                    st.subheader("Prediction Result")
                    
                    if prediction == 'None':
                        st.warning("Review doesn't match target conditions")
                    else:
                        st.success(f"Predicted Condition: {prediction}")
                        
                        # Show top medications
                        st.subheader("Common Medications")
                        top_meds = (data[data['condition'] == prediction]
                                   .groupby('drugName')['rating']
                                   .mean()
                                   .sort_values(ascending=False)
                                   .head(5))
                        
                        for med, rating in top_meds.items():
                            st.write(f"• {med} (Avg rating: {rating:.1f}/10)")
                
                else:
                    st.error("Could not make prediction")
        else:
            st.warning("Please enter a review to analyze")

if __name__ == "__main__":
    main()

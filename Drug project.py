import streamlit as st
import sys
import subprocess
import re

# Package installation check
def install_and_import(package, alias=None):
    try:
        if not alias:
            alias = package.split('.')[0]
        globals()[alias] = __import__(package.split('.')[0])
        if '.' in package:
            for part in package.split('.')[1:]:
                globals()[alias] = getattr(globals()[alias], part)
        return True
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package.split('.')[0]])
        globals()[alias] = __import__(package.split('.')[0])
        if '.' in package:
            for part in package.split('.')[1:]:
                globals()[alias] = getattr(globals()[alias], part)
        return False

# Install and import required packages
required_packages = [
    'pandas',
    'numpy',
    'matplotlib.pyplot:plt',
    'seaborn:sns',
    'wordcloud.WordCloud:WordCloud',
    'textblob.TextBlob:TextBlob'
]

for package_spec in required_packages:
    if ':' in package_spec:
        package, alias = package_spec.split(':')
    else:
        package, alias = package_spec, None
    if not install_and_import(package, alias):
        st.warning(f"Had to install {package}")

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
        # Use read_excel for Excel files and raw string for path
        data = pd.read_excel(r"drugsCom_raw.xlsx")  # Changed to relative path
    except Exception as e:
        st.warning(f"Using sample data because: {str(e)}")
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
    if not data.empty:
        data = data[data['condition'].isin(target_conditions)]
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
    
    return data

# [Rest of your code remains the same until the download button...]

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
            file_name="filtered_drug_reviews.csv",  # Simple filename
            mime="text/csv"
        )

# [Rest of your code...]

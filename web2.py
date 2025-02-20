import os
import pickle
import streamlit as st

# Set page configuration
st.set_page_config(page_title="Diabetes Prediction", layout="wide", page_icon="🧑‍⚕️")

# Load the saved diabetes model safely
diabetes_model_path = r"C:\Users\NAMITHA\OneDrive\Desktop\diabetiesprediction\diabetes_model.sav"

if os.path.exists(diabetes_model_path):
    diabetes_model = pickle.load(open(diabetes_model_path, 'rb'))
    model_loaded = True
else:
    st.error("⚠️ Model file not found! Please check the file path.")
    model_loaded = False

# Page title
st.title('🔬 Diabetes Prediction using ML')

# Get user inputs
col1, col2, col3 = st.columns(3)

with col1:
    Pregnancies = st.text_input('Number of Pregnancies', value="0")

with col2:
    Glucose = st.text_input('Glucose Level', value="0")

with col3:
    BloodPressure = st.text_input('Blood Pressure value', value="0")

with col1:
    SkinThickness = st.text_input('Skin Thickness value', value="0")

with col2:
    Insulin = st.text_input('Insulin Level', value="0")

with col3:
    BMI = st.text_input('BMI value', value="0")

with col1:
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value', value="0")

with col2:
    Age = st.text_input('Age of the Person', value="0")

# Prediction result
diab_diagnosis = ""

# Create prediction button
if st.button('🔍 Diabetes Test Result') and model_loaded:
    try:
        # Convert inputs to float
        user_input = [
            float(Pregnancies), float(Glucose), float(BloodPressure), float(SkinThickness),
            float(Insulin), float(BMI), float(DiabetesPedigreeFunction), float(Age)
        ]

        # Make prediction
        diab_prediction = diabetes_model.predict([user_input])

        # Display result
        diab_diagnosis = '✅ The person is **NOT diabetic**' if diab_prediction[0] == 0 else '⚠️ The person is **DIABETIC**'
        st.success(diab_diagnosis)

    except ValueError:
        st.error("❌ Please enter valid **numeric values** for all fields.")

# Load model accuracy (if available)
accuracy_path = r"C:\Users\NAMITHA\OneDrive\Desktop\diabetiesprediction\diabetes_accuracy.pkl"

if os.path.exists(accuracy_path):
    with open(accuracy_path, 'rb') as f:
        model_accuracy = pickle.load(f)
    accuracy_available = True
else:
    accuracy_available = False

# Show accuracy button
if accuracy_available and st.button('📊 Show Model Accuracy'):
    st.info(f"🔢 **Model Accuracy:** {model_accuracy:.2f}")


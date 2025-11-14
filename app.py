import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import gdown

# -----------------------------
# 🎯 Page Config - MUST BE FIRST
# -----------------------------
st.set_page_config(
    page_title="Heart Disease Predictor", 
    page_icon="❤️", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Try to import with error handling
try:
    import joblib
    import gdown
except ImportError as e:
    st.error(f"Required package missing: {e}")

# -----------------------------
# 🧠 Load Model & Scaler with Compatibility Fix
# -----------------------------
@st.cache_resource
def load_models():
    model_path = "heart_stack_model.joblib"
    scaler_path = "scaler.joblib"
    
    model = None
    scaler = None
    
    try:
        # Download model if not exists
        if not os.path.exists(model_path):
            with st.spinner("📥 Downloading model file..."):
                url = "https://drive.google.com/uc?id=1ikGCWp47yKL-5UbbpY7JH2M79LPeoVLb"
                gdown.download(url, model_path, quiet=True)
        
        # Load scaler first (usually less compatibility issues)
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            st.success(" Scaler loaded successfully!")
        else:
            st.error(f"❌ Scaler file '{scaler_path}' not found.")
        
        # Load model with compatibility handling
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                st.success(" Model loaded successfully!")
            
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
    
    return model, scaler

# Load models
model, scaler = load_models()

# -----------------------------
# 🎨 Custom CSS for Better Styling
# -----------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #ff4b4b;
        text-align: center;
        margin-bottom: 1rem;
    }
    .risk-high {
        background-color: #ffcccc;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff4b4b;
    }
    .risk-low {
        background-color: #ccffcc;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #00cc00;
    }
    .info-box {
        background-color: #e6f3ff;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #0066cc;
    }
    .section-header {
        color: #0066cc;
        border-bottom: 2px solid #0066cc;
        padding-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 💖 App Header
# -----------------------------
st.markdown('<div class="main-header">Heart Disease Risk Assessment</div>', unsafe_allow_html=True)

st.markdown("""
Assess your cardiovascular health risk based on key health indicators and lifestyle factors.
""")

with st.container():
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("**📋 Important Note:** This tool provides a *risk prediction* based on statistical models, not a medical diagnosis. Always consult healthcare professionals for medical advice.")
    st.markdown('</div>', unsafe_allow_html=True)

# Show demo mode if models aren't loaded
if model is None or scaler is None:
    st.warning("🔧 **Demo Mode**: Prediction functionality is currently limited due to model compatibility issues.")
    st.info("""
    **Temporary Workaround:** 
    - The app can still collect and display your health metrics
    - Use the BMI and Blood Pressure indicators for basic health assessment
    - Full prediction will be available after technical maintenance
    """)

# -----------------------------
# 🧍‍♂️ User Inputs with Better Organization
# -----------------------------
st.markdown("---")
st.markdown('<div class="section-header">🩺 Personal & Health Information</div>', unsafe_allow_html=True)

# Personal Information
col1, col2, col3 = st.columns(3)
with col1:
    st.subheader(" Personal")
    age_years = st.number_input("**Age** (years)", min_value=18, max_value=120, value=45, 
                               help="Enter your current age")
    gender = st.selectbox("**Gender**", [1, 2], 
                         format_func=lambda x: "Male" if x == 1 else "Female",
                         help="Select your biological sex")
    
with col2:
    st.subheader(" Physical")
    height = st.slider("**Height** (cm)", min_value=100, max_value=250, value=170, 
                      help="Your height in centimeters")
    weight = st.slider("**Weight** (kg)", min_value=30, max_value=200, value=70, 
                      help="Your weight in kilograms")
    
    # Calculate BMI
    if height > 0:
        bmi = weight / ((height/100) ** 2)
        bmi_category = "Underweight" if bmi < 18.5 else "Normal" if bmi < 25 else "Overweight" if bmi < 30 else "Obese"
        st.metric("**BMI**", f"{bmi:.1f} ({bmi_category})")

with col3:
    st.subheader(" Vital Signs")
    ap_hi = st.number_input("**Systolic BP** (mmHg)", min_value=80, max_value=250, value=120,
                           help="Upper number in blood pressure reading")
    ap_lo = st.number_input("**Diastolic BP** (mmHg)", min_value=50, max_value=150, value=80,
                           help="Lower number in blood pressure reading")
    
    # BP Status
    bp_status = "Normal" if (ap_hi < 120 and ap_lo < 80) else "Elevated" if (ap_hi < 130 and ap_lo < 80) else "High Stage 1" if (ap_hi < 140 and ap_lo < 90) else "High Stage 2" if (ap_hi < 180 and ap_lo < 120) else "Hypertensive Crisis"
    st.metric("**BP Status**", bp_status)

# Lifestyle & Health Markers
st.markdown("---")
st.markdown('<div class="section-header"> Lifestyle & Health Markers</div>', unsafe_allow_html=True)

col4, col5, col6 = st.columns(3)

with col4:
    st.subheader(" Blood Work")
    cholesterol = st.selectbox("**Cholesterol Level**", [1, 2, 3], 
                              format_func=lambda x: ["Normal", "Above Normal", "Well Above Normal"][x-1],
                              help="Your cholesterol level based on recent tests")
    gluc = st.selectbox("**Glucose Level**", [1, 2, 3], 
                       format_func=lambda x: ["Normal", "Above Normal", "Well Above Normal"][x-1],
                       help="Your blood glucose level")

with col5:
    st.subheader(" Habits")
    smoke = st.radio("**Smoking Status**", [0, 1], 
                    format_func=lambda x: " Non-smoker" if x == 0 else " Smoker",
                    help="Regular tobacco use")
    alco = st.radio("**Alcohol Consumption**", [0, 1], 
                   format_func=lambda x: " Non-drinker" if x == 0 else " Drinker",
                   help="Regular alcohol consumption")

with col6:
    st.subheader(" Activity")
    active = st.radio("**Physical Activity**", [1, 0], 
                     format_func=lambda x: " Active" if x == 1 else " Not Active",
                     help="Regular physical exercise")

# -----------------------------
# 🧮 Health Assessment (Demo Mode)
# -----------------------------
st.markdown("---")
st.markdown('<div class="section-header"> Health Assessment</div>', unsafe_allow_html=True)

assess_btn = st.button("** Get Health Assessment**", type="primary", use_container_width=True)

if assess_btn:
    with st.spinner("Analyzing your health profile..."):
        # Simple health assessment based on inputs
        risk_factors = 0
        recommendations = []
        
        # Age factor
        if age_years > 45:
            risk_factors += 1
            recommendations.append(" **Age**: Consider more frequent health check-ups")
        
        # BMI factor
        if bmi >= 25:
            risk_factors += 1
            recommendations.append(" **Weight**: Maintain healthy weight through diet and exercise")
        elif bmi < 18.5:
            risk_factors += 1
            recommendations.append(" **Weight**: Consider nutritional support for healthy weight gain")
        
        # Blood Pressure factor
        if ap_hi >= 130 or ap_lo >= 85:
            risk_factors += 1
            recommendations.append("**Blood Pressure**: Monitor regularly and consult if consistently high")
        
        # Lifestyle factors
        if smoke == 1:
            risk_factors += 2
            recommendations.append(" **Smoking**: Consider smoking cessation programs")
        
        if active == 0:
            risk_factors += 1
            recommendations.append(" **Activity**: Aim for 150 minutes of moderate exercise weekly")
        
        if cholesterol > 1:
            risk_factors += 1
            recommendations.append(" **Cholesterol**: Follow heart-healthy diet low in saturated fats")
        
        if gluc > 1:
            risk_factors += 1
            recommendations.append("**Glucose**: Monitor blood sugar and maintain balanced diet")
        
        # Display assessment
        st.markdown("##  Health Assessment Summary")
        
        if risk_factors <= 2:
            st.markdown('<div class="risk-low">', unsafe_allow_html=True)
            st.markdown("##  Good Health Profile")
            st.markdown(f"**Risk Factors Identified:** {risk_factors} (Low)")
            st.markdown("""
            ###  Keep up the good work!
            Your current health profile shows good indicators. Continue with:
            - Regular physical activity
            - Balanced nutrition
            - Routine health check-ups
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
        elif risk_factors <= 4:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("##  Moderate Health Profile")
            st.markdown(f"**Risk Factors Identified:** {risk_factors} (Moderate)")
            st.markdown("""
            ### 🩺 Areas for Improvement:
            Some lifestyle adjustments could further improve your heart health.
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
        else:
            st.markdown('<div class="risk-high">', unsafe_allow_html=True)
            st.markdown("##  Higher Risk Profile")
            st.markdown(f"**Risk Factors Identified:** {risk_factors} (High)")
            st.markdown("""
            ###  Recommended Actions:
            Consider consulting healthcare professionals for personalized advice.
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Show recommendations
        if recommendations:
            with st.expander(" Personalized Recommendations"):
                for rec in recommendations:
                    st.write(f"• {rec}")
        
        # Technical note
        st.info("""
        **Note:** This is a basic health assessment. For comprehensive heart disease risk prediction, 
        we're working to resolve technical compatibility issues with our machine learning model.
        """)

# -----------------------------
# 📘 Footer
# -----------------------------
st.markdown("---")
footer_col1, footer_col2 = st.columns([3, 1])
with footer_col1:
    st.markdown("""
    **Disclaimer:** This assessment tool is for educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.
    """)
with footer_col2:
    st.markdown("""
    Built by Junayed Bin Karim  
    """)




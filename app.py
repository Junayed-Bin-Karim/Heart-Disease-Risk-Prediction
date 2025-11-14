import streamlit as st
import pandas as pd
import numpy as np
import os

# Try to import optional dependencies
try:
    import joblib
except ImportError:
    st.error("joblib is required but not installed. Please add it to requirements.txt")
    
try:
    import gdown
except ImportError:
    st.error("gdown is required but not installed. Please add it to requirements.txt")

# -----------------------------
# 🎯 Page Config - MUST BE FIRST STREAMLIT COMMAND
# -----------------------------
st.set_page_config(
    page_title="Heart Disease Predictor", 
    page_icon="❤️", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# 🧠 Load Model & Scaler with Error Handling
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
                import gdown
                url = "https://drive.google.com/uc?id=1ikGCWp47yKL-5UbbpY7JH2M79LPeoVLb"
                gdown.download(url, model_path, quiet=True)
        
        # Load model and scaler
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            st.success("✅ Models loaded successfully!")
        else:
            st.error(f"❌ Model files not found. Please ensure {model_path} and {scaler_path} exist.")
            
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

# Show warning if models aren't loaded
if model is None or scaler is None:
    st.warning("⚠️ Models not loaded. Prediction functionality will not work until models are properly loaded.")

# -----------------------------
# 🧍‍♂️ User Inputs with Better Organization
# -----------------------------
st.markdown("---")
st.markdown('<div class="section-header">🩺 Personal & Health Information</div>', unsafe_allow_html=True)

# Personal Information
col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("👤 Personal")
    age_years = st.number_input("**Age** (years)", min_value=18, max_value=120, value=45, 
                               help="Enter your current age")
    gender = st.selectbox("**Gender**", [1, 2], 
                         format_func=lambda x: "Male" if x == 1 else "Female",
                         help="Select your biological sex")
    
with col2:
    st.subheader("📏 Physical")
    height = st.slider("**Height** (cm)", min_value=100, max_value=250, value=170, 
                      help="Your height in centimeters")
    weight = st.slider("**Weight** (kg)", min_value=30, max_value=200, value=70, 
                      help="Your weight in kilograms")
    
    # Calculate BMI
    if height > 0:
        bmi = weight / ((height/100) ** 2)
        st.metric("**BMI**", f"{bmi:.1f}")

with col3:
    st.subheader("💓 Vital Signs")
    ap_hi = st.number_input("**Systolic BP** (mmHg)", min_value=80, max_value=250, value=120,
                           help="Upper number in blood pressure reading")
    ap_lo = st.number_input("**Diastolic BP** (mmHg)", min_value=50, max_value=150, value=80,
                           help="Lower number in blood pressure reading")

# Lifestyle & Health Markers
st.markdown("---")
st.markdown('<div class="section-header">🏃 Lifestyle & Health Markers</div>', unsafe_allow_html=True)

col4, col5, col6 = st.columns(3)

with col4:
    st.subheader("🧪 Blood Work")
    cholesterol = st.selectbox("**Cholesterol Level**", [1, 2, 3], 
                              format_func=lambda x: ["Normal", "Above Normal", "Well Above Normal"][x-1],
                              help="Your cholesterol level based on recent tests")
    gluc = st.selectbox("**Glucose Level**", [1, 2, 3], 
                       format_func=lambda x: ["Normal", "Above Normal", "Well Above Normal"][x-1],
                       help="Your blood glucose level")

with col5:
    st.subheader("🚬 Habits")
    smoke = st.radio("**Smoking Status**", [0, 1], 
                    format_func=lambda x: "🚭 Non-smoker" if x == 0 else "🚬 Smoker",
                    help="Regular tobacco use")
    alco = st.radio("**Alcohol Consumption**", [0, 1], 
                   format_func=lambda x: "🍹 Non-drinker" if x == 0 else "🍷 Drinker",
                   help="Regular alcohol consumption")

with col6:
    st.subheader("🏃 Activity")
    active = st.radio("**Physical Activity**", [1, 0], 
                     format_func=lambda x: "✅ Active" if x == 1 else "❌ Not Active",
                     help="Regular physical exercise")

# -----------------------------
# 🧮 Prediction
# -----------------------------
st.markdown("---")
st.markdown('<div class="section-header">🔍 Risk Assessment</div>', unsafe_allow_html=True)

# Create a prominent prediction button
predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
with predict_col2:
    predict_btn = st.button("**🎯 Calculate My Heart Disease Risk**", 
                          type="primary", 
                          use_container_width=True,
                          disabled=(model is None or scaler is None))

if predict_btn and model is not None and scaler is not None:
    # Show loading spinner
    with st.spinner("🔬 Analyzing your health data..."):
        # Prepare data
        df = pd.DataFrame([{
            'gender': gender,
            'weight': weight,
            'ap_hi': ap_hi,
            'ap_lo': ap_lo,
            'cholesterol': cholesterol,
            'gluc': gluc,
            'smoke': smoke,
            'alco': alco,
            'active': active,
            'age_years': age_years,
            'height_m': height / 100
        }])

        try:
            # Make prediction
            X_scaled = scaler.transform(df)
            prediction = model.predict(X_scaled)[0]
            probability = model.predict_proba(X_scaled)[0][1] * 100

            # -----------------------------
            # 💬 Enhanced Output Result
            # -----------------------------
            st.markdown("## 📊 Assessment Results")
            
            # Progress bar for risk visualization
            st.subheader("Risk Level")
            risk_progress = probability / 100
            st.progress(risk_progress)
            st.caption(f"Estimated Risk Probability: **{probability:.1f}%**")

            # Result with better visual styling
            if prediction == 1:
                st.markdown('<div class="risk-high">', unsafe_allow_html=True)
                st.markdown("## ⚠️ Higher Risk Detected")
                st.markdown(f"**Risk Probability:** `{probability:.1f}%`")
                st.markdown("""
                ### 🩺 Recommended Actions:
                
                **Immediate Steps:**
                - 📞 Consult a healthcare provider soon
                - 🩺 Schedule a comprehensive check-up
                - 📊 Monitor blood pressure regularly
                
                **Lifestyle Changes:**
                - 🥗 Adopt a heart-healthy diet (low salt, sugar, saturated fats)
                - 🚴‍♂️ Increase physical activity (30+ minutes daily)
                - 🚭 Completely avoid tobacco products
                - 🍷 Limit or eliminate alcohol consumption
                - 😴 Ensure 7-8 hours of quality sleep
                - 🧘 Practice stress management techniques
                """)
                st.markdown('</div>', unsafe_allow_html=True)
                
            else:
                st.markdown('<div class="risk-low">', unsafe_allow_html=True)
                st.markdown("## ✅ Lower Risk Profile")
                st.markdown(f"**Risk Probability:** `{probability:.1f}%`")
                st.markdown("""
                ### 💡 Maintenance Tips:
                
                **Keep up the good work! Continue with:**
                - 🏃 Regular physical activity
                - 🥑 Balanced, nutritious diet
                - ⚖️ Healthy weight maintenance
                - 🧘 Stress management practices
                
                **Preventive Care:**
                - 📅 Routine health check-ups annually
                - 🩺 Regular blood pressure monitoring
                - 🧪 Periodic cholesterol and glucose tests
                - 💤 Quality sleep and recovery
                """)
                st.markdown('</div>', unsafe_allow_html=True)

            # Additional health metrics
            with st.expander("📈 View Health Metrics Summary"):
                col_met1, col_met2, col_met3 = st.columns(3)
                with col_met1:
                    st.metric("BMI", f"{bmi:.1f}")
                with col_met2:
                    bp_status = "Normal" if (ap_hi < 130 and ap_lo < 85) else "Monitor"
                    st.metric("Blood Pressure", bp_status)
                with col_met3:
                    st.metric("Physical Activity", "Active" if active == 1 else "Inactive")
                    
        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")

elif predict_btn:
    st.error("❌ Cannot make prediction - models are not loaded properly.")

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

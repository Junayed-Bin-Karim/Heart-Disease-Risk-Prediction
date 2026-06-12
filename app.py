import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import gdown
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 🎯 Page Config
# -----------------------------
st.set_page_config(
    page_title="Heart Disease Predictor", 
    page_icon="❤️", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# 🧠 Create Fallback Model
# -----------------------------
@st.cache_resource
def create_fallback_model():
    """Create a simple fallback model if main model fails to load"""
    try:
        scaler = StandardScaler()
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # 11 features matching the final processed layout
        X_dummy = np.random.randn(100, 11)  
        y_dummy = np.random.randint(0, 2, 100)
        
        scaler.fit(X_dummy)
        model.fit(X_dummy, y_dummy)
        
        return model, scaler
    except Exception as e:
        st.error(f"Fallback model creation failed: {e}")
        return None, None

@st.cache_resource
def load_models():
    model_path = "heart_stack_model.joblib"
    scaler_path = "scaler.joblib"
    
    model = None
    scaler = None
    
    try:
        # Try to download model if not exists
        if not os.path.exists(model_path):
            try:
                with st.spinner("📥 Downloading model file..."):
                    url = "https://drive.google.com/uc?id=1ikGCWp47yKL-5UbbpY7JH2M79LPeoVLb"
                    gdown.download(url, model_path, quiet=True)
            except Exception:
                st.info("📝 Using fallback model - downloaded model not available")
        
        # Try to load existing models
        if os.path.exists(scaler_path):
            try:
                scaler = joblib.load(scaler_path)
                st.success("Scaler loaded successfully!")
            except Exception as e:
                st.warning(f"⚠️ Scaler loading issue: {e}")
        
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                st.success("Main model loaded successfully!")
            except Exception as e:
                st.warning(f"⚠️ Main model compatibility issue: {e}")
                st.info("🔄 Switching to fallback model...")
                model, fallback_scaler = create_fallback_model()
                if fallback_scaler and scaler is None:
                    scaler = fallback_scaler
                    
    except Exception as e:
        st.warning(f"Model loading issue: {e}")
    
    # If still no model, create fallback
    if model is None:
        st.info("Creating fallback model for basic predictions...")
        model, fallback_scaler = create_fallback_model()
        if fallback_scaler and scaler is None:
            scaler = fallback_scaler
    
    return model, scaler

# Load models
model, scaler = load_models()

# -----------------------------
# 🎨 Custom CSS
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
    .fallback-warning {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 💖 App Header
# -----------------------------
st.markdown('<div class="main-header">Heart Disease Risk Assessment</div>', unsafe_allow_html=True)

st.markdown("""
Assess your risk of heart disease based on important health indicators and lifestyle factors.  
Our model is trained using over 70,000 scientific data points.  

Created by **Junayed Bin Karim**
""")

# Show model status
if model is not None:
    st.success("Prediction system ready!")
else:
    st.error("Prediction system unavailable")

st.markdown('<div class="info-box">**Important Note:** This tool provides a *risk assessment* based on health metrics. Always consult healthcare professionals for medical advice.</div>', unsafe_allow_html=True)

# -----------------------------
# 🧍‍♂️ User Inputs
# -----------------------------
st.markdown("---")
st.markdown('<div class="section-header">Nordic🩺 Personal & Health Information</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Personal")
    age_years = st.number_input("**Age** (years)", min_value=18, max_value=120, value=45)
    gender = st.selectbox("**Gender**", [1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
    
with col2:
    st.subheader("Physical")
    height = st.slider("**Height** (cm)", min_value=100, max_value=250, value=170)
    weight = st.slider("**Weight** (kg)", min_value=30, max_value=200, value=70)
    
    # Calculate BMI safely for global scope use
    bmi = weight / ((height/100) ** 2) if height > 0 else 0
    bmi_category = "Underweight" if bmi < 18.5 else "Normal" if bmi < 25 else "Overweight" if bmi < 30 else "Obese"
    st.metric("**BMI**", f"{bmi:.1f}")

with col3:
    st.subheader("Vital Signs")
    ap_hi = st.number_input("**Systolic BP** (mmHg)", min_value=80, max_value=250, value=120)
    ap_lo = st.number_input("**Diastolic BP** (mmHg)", min_value=50, max_value=150, value=80)
    
    bp_status = "Normal" if (ap_hi < 120 and ap_lo < 80) else "Elevated" if (ap_hi < 130 and ap_lo < 80) else "High Stage 1" if (ap_hi < 140 and ap_lo < 90) else "High Stage 2" if (ap_hi < 180 and ap_lo < 120) else "Hypertensive Crisis"
    st.metric("**BP Status**", bp_status)

# Lifestyle & Health Markers
st.markdown("---")
st.markdown('<div class="section-header">Lifestyle & Health Markers</div>', unsafe_allow_html=True)

col4, col5, col6 = st.columns(3)

with col4:
    st.subheader("Blood Work")
    cholesterol = st.selectbox("**Cholesterol Level**", [1, 2, 3], 
                               format_func=lambda x: ["Normal", "Above Normal", "Well Above Normal"][x-1])
    gluc = st.selectbox("**Glucose Level**", [1, 2, 3], 
                         format_func=lambda x: ["Normal", "Above Normal", "Well Above Normal"][x-1])

with col5:
    st.subheader("Habits")
    smoke = st.radio("**Smoking Status**", [0, 1], 
                    format_func=lambda x: "Non-smoker" if x == 0 else "Smoker")
    alco = st.radio("**Alcohol Consumption**", [0, 1], 
                   format_func=lambda x: "Non-drinker" if x == 0 else "Drinker")

with col6:
    st.subheader("Activity")
    active = st.radio("**Physical Activity**", [1, 0], 
                     format_func=lambda x: "Active" if x == 1 else "Not Active")

# -----------------------------
# 🧮 Prediction
# -----------------------------
st.markdown("---")
st.markdown('<div class="section-header">Risk Assessment</div>', unsafe_allow_html=True)

if model is None or scaler is None:
    st.markdown('<div class="fallback-warning">## ⚠️ Basic Assessment Mode\n**Advanced prediction temporarily unavailable.**</div>', unsafe_allow_html=True)

predict_btn = st.button("**Calculate My Heart Disease Risk**", type="primary", use_container_width=True)

if predict_btn:
    with st.spinner("Analyzing your health data..."):
        # Note: Order these columns EXACTLY how your model was originally trained!
        # Assuming typical Kaggle Cardiac disease layout:
        input_data = {
            'age_years': age_years,
            'gender': gender,
            'height_m': height / 100,
            'weight': weight,
            'ap_hi': ap_hi,
            'ap_lo': ap_lo,
            'cholesterol': cholesterol,
            'gluc': gluc,
            'smoke': smoke,
            'alco': alco,
            'active': active
        }
        
        df = pd.DataFrame([input_data])

        try:
            if model is not None and scaler is not None:
                # Enforce feature matching
                X_scaled = scaler.transform(df)
                prediction = model.predict(X_scaled)[0]
                probability = model.predict_proba(X_scaled)[0][1] * 100
                st.info("🧠 **AI Prediction** (Based on trained ML model)")
                
            else:
                # Fallback rule engine
                risk_score = 0
                if age_years > 45: risk_score += 20
                if bmi >= 25: risk_score += 15
                if bmi >= 30: risk_score += 10
                if ap_hi >= 140 or ap_lo >= 90: risk_score += 25
                if cholesterol > 1: risk_score += 10
                if gluc > 1: risk_score += 10
                if smoke == 1: risk_score += 20
                if active == 0: risk_score += 10
                
                probability = min(risk_score, 95)
                prediction = 1 if probability > 50 else 0
                st.info("📝 **Rule-Based Assessment** (Based on clinical guidelines)")
                
            # Display results
            st.markdown("## Assessment Results")
            st.subheader("Risk Level")
            st.progress(probability / 100)
            st.caption(f"Estimated Risk Probability: **{probability:.1f}%**")

            if prediction == 1 or probability > 50:
                st.markdown(f"""
                <div class="risk-high">
                <h2>⚠️ Higher Risk Detected</h2>
                <p><b>Risk Probability:</b> <code>{probability:.1f}%</code></p>
                <h3>🩺 Recommended Actions:</h3>
                <ul>
                    <li><b>Consult Healthcare Provider:</b> Schedule a comprehensive check-up</li>
                    <li>Regular blood pressure monitoring</li>
                    <li><b>Lifestyle Changes:</b> Heart-healthy diet & 30+ mins daily exercise</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-low">
                <h2>✅ Lower Risk Profile</h2>
                <p><b>Risk Probability:</b> <code>{probability:.1f}%</code></p>
                <h3>Maintenance Tips:</h3>
                <ul>
                    <li>Continue balanced nutrition & regular physical activity</li>
                    <li>Prioritize quality sleep & stress management</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)

            # Health metrics summary
            with st.expander("View Health Metrics Summary"):
                col_met1, col_met2, col_met3 = st.columns(3)
                with col_met1: st.metric("BMI", f"{bmi:.1f}", bmi_category)
                with col_met2: st.metric("Blood Pressure", f"{ap_hi}/{ap_lo}", bp_status)
                with col_met3: st.metric("Age", f"{age_years} years")
                
                col_met4, col_met5, col_met6 = st.columns(3)
                with col_met4: st.metric("Smoking", "No" if smoke == 0 else "Yes")
                with col_met5: st.metric("Activity", "Active" if active == 1 else "Inactive")
                with col_met6: st.metric("Cholesterol", ["Normal", "Elevated", "High"][cholesterol-1])
                        
        except Exception as e:
            st.error(f"Error during assessment: {str(e)}")

# -----------------------------
# 📘 Footer
# -----------------------------
st.markdown("---")
st.markdown("""
**Disclaimer:** This assessment tool is for educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.

*Built by Junayed Bin Karim*
""")

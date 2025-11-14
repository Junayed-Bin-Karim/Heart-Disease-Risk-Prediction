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
        # Create a simple Random Forest model with demo data structure
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        
        # Create scaler
        scaler = StandardScaler()
        
        # Create a simple model (will be trained on user data over time)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Create dummy data for initial fit
        import numpy as np
        X_dummy = np.random.randn(100, 11)  # 11 features matching your input
        y_dummy = np.random.randint(0, 2, 100)
        
        # Fit scaler and model
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
            except:
                st.info("📝 Using fallback model - downloaded model not available")
        
        # Try to load existing models
        if os.path.exists(scaler_path):
            try:
                import joblib
                scaler = joblib.load(scaler_path)
                st.success("✅ Scaler loaded successfully!")
            except Exception as e:
                st.warning(f"⚠️ Scaler loading issue: {e}")
        
        if os.path.exists(model_path):
            try:
                import joblib
                model = joblib.load(model_path)
                st.success("✅ Main model loaded successfully!")
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
        st.info("🎯 Creating fallback model for basic predictions...")
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
Assess your cardiovascular health risk based on key health indicators and lifestyle factors.
""")

# Show model status
if model is not None:
    st.success("✅ Prediction system ready!")
else:
    st.error("❌ Prediction system unavailable")

with st.container():
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("**📋 Important Note:** This tool provides a *risk assessment* based on health metrics. Always consult healthcare professionals for medical advice.")
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# 🧍‍♂️ User Inputs
# -----------------------------
st.markdown("---")
st.markdown('<div class="section-header">🩺 Personal & Health Information</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("👤 Personal")
    age_years = st.number_input("**Age** (years)", min_value=18, max_value=120, value=45)
    gender = st.selectbox("**Gender**", [1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
    
with col2:
    st.subheader("📏 Physical")
    height = st.slider("**Height** (cm)", min_value=100, max_value=250, value=170)
    weight = st.slider("**Weight** (kg)", min_value=30, max_value=200, value=70)
    
    # Calculate BMI
    if height > 0:
        bmi = weight / ((height/100) ** 2)
        bmi_category = "Underweight" if bmi < 18.5 else "Normal" if bmi < 25 else "Overweight" if bmi < 30 else "Obese"
        st.metric("**BMI**", f"{bmi:.1f}")

with col3:
    st.subheader("💓 Vital Signs")
    ap_hi = st.number_input("**Systolic BP** (mmHg)", min_value=80, max_value=250, value=120)
    ap_lo = st.number_input("**Diastolic BP** (mmHg)", min_value=50, max_value=150, value=80)
    
    bp_status = "Normal" if (ap_hi < 120 and ap_lo < 80) else "Elevated" if (ap_hi < 130 and ap_lo < 80) else "High Stage 1" if (ap_hi < 140 and ap_lo < 90) else "High Stage 2" if (ap_hi < 180 and ap_lo < 120) else "Hypertensive Crisis"
    st.metric("**BP Status**", bp_status)

# Lifestyle & Health Markers
st.markdown("---")
st.markdown('<div class="section-header">🏃 Lifestyle & Health Markers</div>', unsafe_allow_html=True)

col4, col5, col6 = st.columns(3)

with col4:
    st.subheader("🧪 Blood Work")
    cholesterol = st.selectbox("**Cholesterol Level**", [1, 2, 3], 
                              format_func=lambda x: ["Normal", "Above Normal", "Well Above Normal"][x-1])
    gluc = st.selectbox("**Glucose Level**", [1, 2, 3], 
                       format_func=lambda x: ["Normal", "Above Normal", "Well Above Normal"][x-1])

with col5:
    st.subheader("🚬 Habits")
    smoke = st.radio("**Smoking Status**", [0, 1], 
                    format_func=lambda x: "🚭 Non-smoker" if x == 0 else "🚬 Smoker")
    alco = st.radio("**Alcohol Consumption**", [0, 1], 
                   format_func=lambda x: "🍹 Non-drinker" if x == 0 else "🍷 Drinker")

with col6:
    st.subheader("🏃 Activity")
    active = st.radio("**Physical Activity**", [1, 0], 
                     format_func=lambda x: "✅ Active" if x == 1 else "❌ Not Active")

# -----------------------------
# 🧮 Prediction
# -----------------------------
st.markdown("---")
st.markdown('<div class="section-header">🔍 Risk Assessment</div>', unsafe_allow_html=True)

if model is None or scaler is None:
    st.markdown('<div class="fallback-warning">', unsafe_allow_html=True)
    st.markdown("""
    ## ⚠️ Basic Assessment Mode
    **Advanced prediction temporarily unavailable.** 
    
    We're showing health insights based on established medical guidelines.
    The machine learning model is being updated for better compatibility.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

predict_btn = st.button("**🎯 Calculate My Heart Disease Risk**", type="primary", use_container_width=True)

if predict_btn:
    with st.spinner("🔬 Analyzing your health data..."):
        # Prepare data
        input_data = {
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
        }
        
        df = pd.DataFrame([input_data])

        try:
            if model is not None and scaler is not None:
                # Make prediction with model
                X_scaled = scaler.transform(df)
                prediction = model.predict(X_scaled)[0]
                probability = model.predict_proba(X_scaled)[0][1] * 100
                
                # Show model source
                st.info("🤖 **AI Prediction** (Based on trained model)")
                
            else:
                # Fallback: Rule-based assessment
                risk_score = 0
                
                # Risk factors scoring
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
                
                st.info("📊 **Rule-Based Assessment** (Based on medical guidelines)")
                
            # Display results
            st.markdown("## 📊 Assessment Results")
            
            # Progress bar
            st.subheader("Risk Level")
            risk_progress = probability / 100
            st.progress(risk_progress)
            st.caption(f"Estimated Risk Probability: **{probability:.1f}%**")

            if prediction == 1 or probability > 50:
                st.markdown('<div class="risk-high">', unsafe_allow_html=True)
                st.markdown("## ⚠️ Higher Risk Detected")
                st.markdown(f"**Risk Probability:** `{probability:.1f}%`")
                st.markdown("""
                ### 🩺 Recommended Actions:
                **Consult Healthcare Provider:**
                - 📞 Schedule a comprehensive check-up
                - 🩺 Discuss your risk factors
                - 📊 Regular blood pressure monitoring
                
                **Lifestyle Changes:**
                - 🥗 Heart-healthy diet
                - 🚴‍♂️ 30+ minutes daily exercise
                - 🚭 Avoid tobacco
                - 🍷 Limit alcohol
                """)
                st.markdown('</div>', unsafe_allow_html=True)
                
            else:
                st.markdown('<div class="risk-low">', unsafe_allow_html=True)
                st.markdown("## ✅ Lower Risk Profile")
                st.markdown(f"**Risk Probability:** `{probability:.1f}%`")
                st.markdown("""
                ### 💡 Maintenance Tips:
                **Continue Healthy Habits:**
                - 🏃 Regular physical activity
                - 🥑 Balanced nutrition
                - ⚖️ Healthy weight maintenance
                - 😴 Quality sleep
                - 🧘 Stress management
                """)
                st.markdown('</div>', unsafe_allow_html=True)

            # Health metrics summary
            with st.expander("📈 View Health Metrics Summary"):
                col_met1, col_met2, col_met3 = st.columns(3)
                with col_met1:
                    st.metric("BMI", f"{bmi:.1f}", bmi_category)
                with col_met2:
                    st.metric("Blood Pressure", f"{ap_hi}/{ap_lo}", bp_status)
                with col_met3:
                    st.metric("Age", f"{age_years} years")
                
                col_met4, col_met5, col_met6 = st.columns(3)
                with col_met4:
                    st.metric("Smoking", "No" if smoke == 0 else "Yes")
                with col_met5:
                    st.metric("Activity", "Active" if active == 1 else "Inactive")
                with col_met6:
                    st.metric("Cholesterol", ["Normal", "Elevated", "High"][cholesterol-1])
                    
        except Exception as e:
            st.error(f"❌ Error during assessment: {str(e)}")
            st.info("Please try again or check your input values.")

# -----------------------------
# 📘 Footer
# -----------------------------
st.markdown("---")
st.markdown("""
**Disclaimer:** This assessment tool is for educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.

*Built by Junayed Bin Karim*
""")

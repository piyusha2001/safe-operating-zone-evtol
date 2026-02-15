import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

st.set_page_config(
    page_title="Project Ūrdhyuth | Safety Advisor",
    page_icon="🚁",
    layout="wide"
)


@st.cache_resource
def load_assets():
    model = joblib.load('urdhyuth_safety_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
except FileNotFoundError:
    st.error("Error: Model files not found. Please make sure 'urdhyuth_safety_model.pkl' and 'scaler.pkl' are in the same folder.")
    st.stop()


st.sidebar.header("⚙️ Mission Parameters")
st.sidebar.markdown("Adjust simulation values below:")


payload = st.sidebar.slider("📦 Payload (kg)", 0, 800, 500, help="Max Limit: 600kg")
altitude = st.sidebar.slider("ALT Altitude (m)", 0, 800, 150, help="Service Ceiling: 610m")
soc = st.sidebar.slider("🔋 Battery SOC (%)", 0, 100, 80, help="Reserve Required: 30%")
ambient_temp = st.sidebar.slider("🌡️ Ambient Temp (°C)", -20, 60, 25, help="Ops Limit: 45°C")
wind_speed = st.sidebar.slider("💨 Wind Speed (m/s)", 0, 35, 5, help="Safe limit approx 15 m/s")


st.sidebar.markdown("---")
st.sidebar.subheader("📡 Telemetry Readings")
climb_rate = st.sidebar.slider("Vertical Speed (m/s)", -10.0, 10.0, 0.0)
battery_temp = st.sidebar.slider("Battery Temp (°C)", 20, 90, 35)
thrust_demand = st.sidebar.slider("Motor Power Demand (%)", 0, 100, 60)

st.title("🚁 Project Ūrdhyuth: AI Safety Advisor")
st.markdown("Real-time Flight Envelope Prediction System for eVTOL.")


input_data = pd.DataFrame({
    'altitude_m': [altitude],
    'wind_speed_m_s': [wind_speed],
    'payload_kg': [payload],
    'ambient_temp_c': [ambient_temp],
    'soc_percent': [soc],
    'battery_temp_c': [battery_temp],
    'climb_rate_m_s': [climb_rate],
    'thrust_demand_percent': [thrust_demand]
})


prediction_index = model.predict(input_data)[0]
probabilities = model.predict_proba(input_data)[0]


classes = ['Marginal', 'Safe', 'Unsafe']
prediction_label = classes[prediction_index]



col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Flight Envelope Monitor")
    
    if prediction_label == "Unsafe":
        gauge_value = 15
        gauge_color = "red"
        status_text = "UNSAFE"
    elif prediction_label == "Marginal":
        gauge_value = 50
        gauge_color = "orange"
        status_text = "MARGINAL"
    else:
        gauge_value = 85
        gauge_color = "green"
        status_text = "SAFE"

    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = gauge_value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Safety Index", 'font': {'size': 24}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "black", 'thickness': 0.02},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 33], 'color': '#ff4b4b'},
                {'range': [33, 66], 'color': '#ffa421'},
                {'range': [66, 100], 'color': '#21c354'}
            ],
            'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': gauge_value}
        }
    ))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"<h2 style='text-align: center; color: {gauge_color};'>{status_text}</h2>", unsafe_allow_html=True)

    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Payload Margin", f"{600 - payload} kg", delta_color="normal", help="PRD Limit: 600kg")
    m2.metric("Temp Margin", f"{45 - ambient_temp} °C", delta_color="inverse", help="PRD Range: -10 to 45°C")
    m3.metric("Batt Reserve", f"{soc}%", delta_color="off" if soc >= 30 else "inverse", help="PRD Min Reserve: 30%")
    m4.metric("Alt Margin", f"{610 - altitude} m", delta_color="normal", help="PRD Ceiling: 610m")

with col2:
    st.subheader("AI Confidence")
    prob_df = pd.DataFrame(probabilities, index=classes, columns=["Probability"])
    st.bar_chart(prob_df)
 
    st.subheader("Active Alerts")
    
    active_alerts = False

    if climb_rate > 8.0:
        st.warning("⚠️ **HIGH CLIMB RATE:** Excessive Motor Stress (>8 m/s)")
        active_alerts = True
    
   
    elif climb_rate < -5.0:
        st.error("🚨 **SINK RATE:** Risk of Vortex Ring State (<-5 m/s)")
        active_alerts = True

    if altitude < 30 and climb_rate < -3.0:
         st.error("📉 **HARD LANDING:** Reduce Descent Rate Immediately")
         active_alerts = True
         
    
    if payload > 600:
        st.error("🚨 **OVERWEIGHT:** Exceeds 600kg Design Limit")
        active_alerts = True

    
    if altitude > 610:
        st.error("🚨 **ALTITUDE:** Exceeds Service Ceiling (610m)")
        active_alerts = True

    
    if ambient_temp > 45:
        st.error("🚨 **HEAT CRITICAL:** Exceeds +45°C Limit")
        active_alerts = True
    elif ambient_temp < -10:
        st.error("❄️ **FREEZING:** Below -10°C Min Limit")
        active_alerts = True

    if soc < 30:
        st.warning("⚠️ **RESERVE:** Battery below 30% mandate")
        active_alerts = True

    if wind_speed > 15:
        st.warning("⚠️ **WIND:** High Turbulence Risk")
        active_alerts = True
    
    if not active_alerts:
        st.success("✅ Operations within PRD Spec")

with st.expander("ℹ️ How does the AI decide?"):
    st.write("""
    The system uses a **Random Forest Classifier** trained on 20,000 physics-based flight scenarios.
    It analyzes the combination of environmental factors (Wind, Temp) and Vehicle State (Load, SOC) 
    to determine if the aircraft remains within the 'Safe Operating Zone' defined in the Project Ūrdhyuth PRD.
    """)
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os

# --- Page Config & Styling ---
st.set_page_config(page_title="Weather Predictor Pro", page_icon="🌡️", layout="wide")

# Custom CSS for UI Enhancement
st.markdown(f"""
    <style>
    /* Main Background and Sidebar */
    .stApp {{
        background-color: #A25AE1;
    }}
    .css-1d391kg {{
        background-color: #054A91;
    }}
    
    /* Headers */
    h1, h2, h3 {{
        color: #054A91;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }}

    /* Prediction Box Styling */
    .prediction-card {{
        background-color: #054A91;
        padding: 30px;
        border-radius: 15px;
        color: #FFFFFF;
        text-align: center;
        box-shadow: 5px 5px 15px rgba(0,0,0,0.2);
        border-left: 10px solid #F17300;
    }}
    .prediction-value {{
        font-size: 50px;
        font-weight: bold;
        color: #F17300;
    }}
    
    /* Button and Sliders */
    .stSlider > div > div > div > div {{
        background-color: #3E7CB1;
    }}
    </style>
    """, unsafe_allow_html=True)

# --- App Title ---
st.markdown("<h1 style='text-align: center;'> Weather Temperature Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #3E7CB1;'>Using Linear Regression for Accurate Forecasting</p>", unsafe_allow_html=True)
st.divider()

# --- Dataset Load ---
file_name = 'Weather_Data.csv'

@st.cache_data
def load_and_preprocess_data(file):
    if not os.path.exists(file):
        return None
    df = pd.read_csv(file)
    df = df.dropna(subset=['MinTemp', 'MaxTemp', 'Rainfall', 'Humidity3pm', 'Pressure3pm', 'RainToday'])
    df['RainToday'] = df['RainToday'].map({'Yes': 1, 'No': 0})
    return df

df = load_and_preprocess_data(file_name)

if df is None:
    st.error(f"❌ '{file_name}' file-ti GitHub-e pawa jachhe na!")
    st.stop()

# --- Model Training ---
X = df[['MinTemp', 'Rainfall', 'Humidity3pm', 'Pressure3pm', 'RainToday']]
y = df['MaxTemp']
model = LinearRegression()
model.fit(X, y)

# --- Layout: Columns for Input and Output ---
col_in, col_out = st.columns([1, 1.2], gap="large")

with col_in:
    st.markdown("### 🛠️ Input Parameters")
    with st.container():
        min_t = st.slider('Minimum Temperature (°C)', float(df.MinTemp.min()), float(df.MinTemp.max()), float(df.MinTemp.mean()))
        rain = st.slider('Rainfall (mm)', float(df.Rainfall.min()), float(df.Rainfall.max()), float(df.Rainfall.mean()))
        hum = st.slider('Humidity at 3pm (%)', float(df.Humidity3pm.min()), float(df.Humidity3pm.max()), float(df.Humidity3pm.mean()))
        press = st.slider('Pressure at 3pm (hPa)', float(df.Pressure3pm.min()), float(df.Pressure3pm.max()), float(df.Pressure3pm.mean()))
        rt = st.selectbox('Rain Today?', ['No', 'Yes'])
        rt_val = 1 if rt == 'Yes' else 0

with col_out:
    st.markdown("### 🎯 Prediction Analysis")
    user_input = pd.DataFrame({
        'MinTemp': [min_t], 'Rainfall': [rain], 
        'Humidity3pm': [hum], 'Pressure3pm': [press], 'RainToday': [rt_val]
    })
    
    prediction = model.predict(user_input)[0]
    
    # Stunning Prediction Card
    st.markdown(f"""
        <div class="prediction-card">
            <h3>Predicted Maximum Temperature</h3>
            <div class="prediction-value">{prediction:.2f} °C</div>
            <p style="color: #DBE4EE;">Based on current meteorological inputs</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.write("")
    st.info("ℹ️ Tip: Adjust the sliders on the left to see how temperature changes in real-time.")

# --- Bottom Info ---
st.divider()
st.markdown(f"<p style='text-align: center; color: #054A91;'>Model Accuracy: <b>{model.score(X, y)*100:.2f}%</b> | Dataset Size: {len(df)} rows</p>", unsafe_allow_html=True)




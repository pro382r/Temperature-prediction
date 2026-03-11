import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os

# --- Page Config ---
st.set_page_config(page_title="Weather Pro", page_icon="🌡️", layout="wide")

# Custom CSS for Adaptive UI
st.markdown(f"""
    <style>
    /* Main Background: Slate Teal */
    .stApp {{
        background-color: #2A4D69;
    }}
    
    /* Global Text Color to Light Blue/White */
    h1, h2, h3, p, span, label {{
        color: #DBE4EE !important;
    }}

    /* Prediction Card */
    .prediction-card {{
        background-color: rgba(255, 255, 255, 0.1); /* Transparent glass effect */
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        border: 2px solid #F17300;
        backdrop-filter: blur(10px);
    }}
    
    .prediction-value {{
        font-size: 55px;
        font-weight: 900;
        color: #F17300;
    }}

    /* Sidebar Styling */
    [data-testid="stSidebar"] {{
        background-color: rgba(0, 0, 0, 0.2);
    }}
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'> Weather Predictor</h1>", unsafe_allow_html=True)
st.divider()

# --- Load Data ---
file_name = 'Weather_Data.csv'
@st.cache_data
def load_data(file):
    if not os.path.exists(file): return None
    df = pd.read_csv(file).dropna(subset=['MinTemp', 'MaxTemp', 'Rainfall', 'Humidity3pm', 'Pressure3pm', 'RainToday'])
    df['RainToday'] = df['RainToday'].map({'Yes': 1, 'No': 0})
    return df

df = load_data(file_name)

if df is None:
    st.error("Dataset not found!")
    st.stop()

# --- Model ---
X = df[['MinTemp', 'Rainfall', 'Humidity3pm', 'Pressure3pm', 'RainToday']]
y = df['MaxTemp']
model = LinearRegression().fit(X, y)

# --- Layout ---
col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.markdown("### 🛠️ Adjust Features")
    min_t = st.slider('Min Temp', float(df.MinTemp.min()), float(df.MinTemp.max()), float(df.MinTemp.mean()))
    rain = st.slider('Rainfall', float(df.Rainfall.min()), float(df.Rainfall.max()), float(df.Rainfall.mean()))
    hum = st.slider('Humidity', float(df.Humidity3pm.min()), float(df.Humidity3pm.max()), float(df.Humidity3pm.mean()))
    press = st.slider('Pressure', float(df.Pressure3pm.min()), float(df.Pressure3pm.max()), float(df.Pressure3pm.mean()))
    rt_val = 1 if st.selectbox('Rain Today?', ['No', 'Yes']) == 'Yes' else 0

with col2:
    st.markdown("### 🎯 Result")
    input_data = pd.DataFrame([[min_t, rain, hum, press, rt_val]], columns=X.columns)
    prediction = model.predict(input_data)[0]
    
    st.markdown(f"""
        <div class="prediction-card">
            <h3>Estimated Max Temp</h3>
            <div class="prediction-value">{prediction:.2f} °C</div>
            <p>Model Accuracy: {model.score(X, y)*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

st.divider()
st.markdown("<p style='text-align: center;'>Developed with ❤️ for Temperature Prediction Project</p>", unsafe_allow_html=True)


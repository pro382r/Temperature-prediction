import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os

# --- Page Config ---
st.set_page_config(page_title="Temperature Predictor", page_icon="🌡️")

st.title("🌡️ Weather Temperature Prediction App")
st.write("Linear Regression bebohar kore Max Temperature predict kora hocche.")

# --- Dataset Load (With Error Handling) ---
file_name = 'Weather_Data.csv'

@st.cache_data
def load_and_preprocess_data(file):
    if not os.path.exists(file):
        return None
    
    df = pd.read_csv(file)
    # Basic Preprocessing
    df = df.dropna(subset=['MinTemp', 'MaxTemp', 'Rainfall', 'Humidity3pm', 'Pressure3pm', 'RainToday'])
    df['RainToday'] = df['RainToday'].map({'Yes': 1, 'No': 0})
    return df

df = load_and_preprocess_data(file_name)

if df is None:
    st.error(f"❌ '{file_name}' file-ti GitHub-e pawa jachhe na!")
    st.info("Solution: GitHub-e jekhane app.py ache, thik shei folder-e Weather_Data.csv file-ti upload koro.")
    st.stop()

# --- Model Training ---
X = df[['MinTemp', 'Rainfall', 'Humidity3pm', 'Pressure3pm', 'RainToday']]
y = df['MaxTemp']
model = LinearRegression()
model.fit(X, y)

# --- Sidebar Inputs ---
st.sidebar.header("User Input (Features)")

def get_user_inputs():
    min_t = st.sidebar.slider('Minimum Temperature (°C)', float(df.MinTemp.min()), float(df.MinTemp.max()), float(df.MinTemp.mean()))
    rain = st.sidebar.slider('Rainfall (mm)', float(df.Rainfall.min()), float(df.Rainfall.max()), float(df.Rainfall.mean()))
    hum = st.sidebar.slider('Humidity at 3pm (%)', float(df.Humidity3pm.min()), float(df.Humidity3pm.max()), float(df.Humidity3pm.mean()))
    press = st.sidebar.slider('Pressure at 3pm (hPa)', float(df.Pressure3pm.min()), float(df.Pressure3pm.max()), float(df.Pressure3pm.mean()))
    rt = st.sidebar.selectbox('Rain Today?', ['No', 'Yes'])
    rt_val = 1 if rt == 'Yes' else 0
    
    return pd.DataFrame({
        'MinTemp': [min_t],
        'Rainfall': [rain],
        'Humidity3pm': [hum],
        'Pressure3pm': [press],
        'RainToday': [rt_val]
    })

user_df = get_user_inputs()

# --- Prediction Result ---
st.subheader("Prediction Results")
col1, col2 = st.columns(2)

prediction = model.predict(user_df)

with col1:
    st.markdown("### Input Data")
    st.write(user_df)

with col2:
    st.markdown("### Predicted Max Temp")
    st.success(f"## {prediction[0]:.2f} °C")

# --- Visual Comparison (Optional) ---
st.divider()
st.subheader("Model Insights")
st.write(f"Dataset-e total record ache: {len(df)}")

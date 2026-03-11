import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# App Title
st.title("🌡️ Temperature Prediction App")
st.write("Ekhane input diye check koro shorboccho tapmatra (MaxTemp) koto hote pare.")

# Load Data (Cache kora hoy jate bar bar load na hoy)
@st.cache_data
def load_data():
    df = pd.read_csv('Weather_Data.csv')
    df = df.dropna()
    df['RainToday'] = df['RainToday'].map({'Yes': 1, 'No': 0})
    return df

df = load_data()
# Jodi file-ti GitHub-er main folder-e thake:
df = pd.read_csv('Weather_Data.csv')

# Model Train kora (User input er agei background e hobe)
X = df[['MinTemp', 'Rainfall', 'Humidity3pm', 'Pressure3pm', 'RainToday']]
y = df['MaxTemp']
model = LinearRegression()
model.fit(X, y)

# --- Sidebar Input ---
st.sidebar.header("User Input Parameters")

def user_input_features():
    min_temp = st.sidebar.slider('Minimum Temperature', float(df.MinTemp.min()), float(df.MinTemp.max()), float(df.MinTemp.mean()))
    rainfall = st.sidebar.slider('Rainfall', float(df.Rainfall.min()), float(df.Rainfall.max()), float(df.Rainfall.mean()))
    humidity = st.sidebar.slider('Humidity at 3pm', float(df.Humidity3pm.min()), float(df.Humidity3pm.max()), float(df.Humidity3pm.mean()))
    pressure = st.sidebar.slider('Pressure at 3pm', float(df.Pressure3pm.min()), float(df.Pressure3pm.max()), float(df.Pressure3pm.mean()))
    rain_today = st.sidebar.selectbox('Rain Today?', ('Yes', 'No'))
    rain_today_val = 1 if rain_today == 'Yes' else 0
    
    data = {
        'MinTemp': min_temp,
        'Rainfall': rainfall,
        'Humidity3pm': humidity,
        'Pressure3pm': pressure,
        'RainToday': rain_today_val
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- Display Prediction ---
st.subheader('Tomar dewa Input:')
st.write(input_df)

prediction = model.predict(input_df)

st.subheader('Predicted Maximum Temperature:')

st.write(f"### {prediction[0]:.2f} °C")

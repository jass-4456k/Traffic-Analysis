import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_artifacts():
    model = joblib.load('traffic_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
    return model, scaler, feature_names

try:
    model, scaler, feature_names = load_artifacts()
except FileNotFoundError:
    st.error("Model artifacts not found. Please run 'python train.py' first.")
    st.stop()

st.title("🚦 Metro Traffic Volume Predictor")
st.write("Adjust the parameters below to predict the hourly traffic volume (vehicles per hour).")

col1, col2 = st.columns(2)

with col1:
    st.header("🕒 Time Features")
    hour = st.slider("Hour of Day (0-23)", 0, 23, 12)
    
    day_mapping = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
    day = st.selectbox("Day of Week", list(day_mapping.keys()))
    day_idx = day_mapping[day]
    is_weekend = 1 if day_idx >= 5 else 0
    
    month = st.slider("Month (1-12)", 1, 12, 6)

with col2:
    st.header("🌤️ Weather Features")
    temp_c = st.number_input("Temperature (°C)", value=20.0)
    temp_k = temp_c + 273.15 
    
    rain = st.number_input("Rain in last hour (mm)", min_value=0.0, value=0.0)
    snow = st.number_input("Snow in last hour (mm)", min_value=0.0, value=0.0)
    clouds = st.slider("Cloud Cover (%)", 0, 100, 50)
    
    weather_options = ['Clear', 'Clouds', 'Drizzle', 'Fog', 'Haze', 'Mist', 'Rain', 'Smoke', 'Snow', 'Squall', 'Thunderstorm']
    weather_main = st.selectbox("Weather Condition", weather_options)

if st.button("Predict Traffic Volume"):
    
    # --- VALIDATION CHECKS ---
    valid_input = True

    # 1. Check if temperature exceeds 50°C
    if temp_c > 50.0:
        st.error("❌ **Invalid Input:** Temperature cannot exceed 50°C.")
        valid_input = False
        
    # 2. Check if Snow condition is selected or snow volume is reported when temperature is above 5°C
    # (Using 5°C as a realistic upper limit for snow to exist/fall)
    elif (weather_main == 'Snow' or snow > 0.0) and temp_c > 5.0:
        st.error(f"❌ **Weather Mismatch:** It cannot snow at {temp_c}°C. Please adjust the temperature or weather condition.")
        valid_input = False

    # --- PREDICTION LOGIC ---
    if valid_input:
        input_data = {
            'temp': temp_k,
            'rain_1h': rain,
            'snow_1h': snow,
            'clouds_all': clouds,
            'hour': hour,
            'day_of_week': day_idx,
            'month': month,
            'is_weekend': is_weekend
        }

        for col in feature_names:
            if col.startswith('weather_main_'):
                expected_weather = col.replace('weather_main_', '')
                input_data[col] = 1 if weather_main == expected_weather else 0

        input_df = pd.DataFrame([input_data])
        input_df = input_df.reindex(columns=feature_names, fill_value=0)

        scale_cols = ['temp', 'rain_1h', 'snow_1h', 'clouds_all']
        input_df[scale_cols] = scaler.transform(input_df[scale_cols])

        prediction = model.predict(input_df)[0]
        
        st.success(f"### Predicted Traffic Volume: **{int(prediction)}** vehicles/hour")
        
        if prediction >= 4500:
            st.error("⚠️ Heavy Congestion Expected.")
        elif prediction >= 2500:
            st.warning("🚗 Moderate Traffic Flow.")
        else:
            st.info("✅ Smooth Traffic / Light Volume.")
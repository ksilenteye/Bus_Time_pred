import streamlit as st
import numpy as np
from joblib import load

model = load('model.joblib')
label_encoder_day = load('day_encoder.joblib')
label_encoder_route = load('route_encoder.joblib')

st.title("Trip End Time Prediction")

Sch_Start_Hour = st.number_input("Scheduled Start Hour", min_value=0, max_value=23)
Sch_Start_Minute = st.number_input("Scheduled Start Minute", min_value=0, max_value=59)
Operated_Day = st.selectbox("Operated Day", label_encoder_day.classes_)
Route_Name = st.selectbox("Route Name", label_encoder_route.classes_)

Operated_Day_Encoded = label_encoder_day.transform([Operated_Day])[0]
Route_Name_Encoded = label_encoder_route.transform([Route_Name])[0]

input_features = np.array([[Sch_Start_Hour, Sch_Start_Minute, Operated_Day_Encoded, Route_Name_Encoded]])

if st.button("Predict"):
    predicted_time_minutes = model.predict(input_features)[0]
    
    end_hour = Sch_Start_Hour + int(predicted_time_minutes // 60)
    end_minute = Sch_Start_Minute + int(predicted_time_minutes % 60)
    
    if end_minute >= 60:
        end_hour += 1
        end_minute -= 60
    
    if end_hour >= 24:
        end_hour -= 24

        end_hour = -end_hour
    
    st.write(f"Predicted Trip End Time (in minutes): {predicted_time_minutes:.2f} minutes")
    st.write(f"Scheduled Start Time: {Sch_Start_Hour:02d}:{Sch_Start_Minute:02d}")
    
    if end_hour < 0:
        st.write(f"Estimated Arrival Time: -{abs(end_hour):02d}:{abs(end_minute):02d} (next day)")
    else:
        st.write(f"Estimated Arrival Time: {end_hour:02d}:{end_minute:02d}")

    decoded_day = label_encoder_day.inverse_transform([Operated_Day_Encoded])[0]
    decoded_route = label_encoder_route.inverse_transform([Route_Name_Encoded])[0]
    st.write(f"Operated Day: {decoded_day}")
    st.write(f"Route Name: {decoded_route}")

import streamlit as st
import pandas as pd
import joblib

model = joblib.load('tip_model.pkl')
scaler = joblib.load('sc.pkl')
st.title('Waiter Tip Prediction App')
# Input fields
total_bill = st.number_input("Total Bill ($)", min_value=0.0, value=20.0)
sex = st.selectbox("Sex", ["Male", "Female"])
smoker = st.selectbox("Smoker", ["Yes", "No"])
day = st.selectbox("Day of Week", ["Thur", "Fri", "Sat", "Sun"])
time = st.selectbox("Time", ["Lunch", "Dinner"])
size = st.number_input("Number of People", min_value=1, value=2)

# Encode categorical inputs
sex_val = 1 if sex=="Male" else 0
smoker_val = 1 if smoker=="Yes" else 0
day_mapping = {"Thur":0, "Fri":1, "Sat":2, "Sun":3}
day_val = day_mapping[day]
time_val = 1 if time=="Dinner" else 0

# Prepare input data
input_data = pd.DataFrame([[total_bill, sex_val, smoker_val, day_val, time_val, size]],
                          columns=["total_bill","sex","smoker","day","time","size"])

# Scale features
input_scaled = scaler.transform(input_data)

# Predict tip
predicted_tip = model.predict(input_scaled)[0]

st.success(f"Predicted Tip: ${predicted_tip:.2f}")

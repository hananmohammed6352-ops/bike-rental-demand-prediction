import streamlit as st
import pandas as pd
import gzip
import pickle

st.title("Bike Rental Demand Prediction")

# Load model
with gzip.open("model.pkl.gz", "rb") as f:
    model = pickle.load(f)

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load feature columns
with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

st.write("Enter the details below:")

# Numerical inputs
temp = st.number_input("Temperature", value=20.0)
hum = st.number_input("Humidity", value=50.0)
windspeed = st.number_input("Windspeed", value=10.0)

# Categorical inputs
season = st.selectbox(
    "Season",
    ["Fall", "Spring", "Summer", "Winter"]
)

yr = st.selectbox(
    "Year",
    [0, 1]
)

mnth = st.selectbox(
    "Month",
    list(range(1, 13))
)

hr = st.selectbox(
    "Hour",
    list(range(24))
)

holiday = st.selectbox(
    "Holiday",
    [0, 1]
)

weekday = st.selectbox(
    "Weekday",
    list(range(7))
)

workingday = st.selectbox(
    "Working Day",
    [0, 1]
)

weathersit = st.selectbox(
    "Weather Situation",
    ["Clear", "Heavy Rain/Snow", "Light Rain/Snow", "Mist"]
)

if st.button("Predict Bike Rental Demand"):

    # Create input DataFrame
    input_data = pd.DataFrame({
        "season": [season],
        "yr": [yr],
        "mnth": [mnth],
        "hr": [hr],
        "holiday": [holiday],
        "weekday": [weekday],
        "workingday": [workingday],
        "weathersit": [weathersit],
        "temp": [temp],
        "hum": [hum],
        "windspeed": [windspeed]
    })

    # One-hot encoding
    input_encoded = pd.get_dummies(
        input_data,
        columns=[
            "season",
            "yr",
            "mnth",
            "hr",
            "holiday",
            "weekday",
            "workingday",
            "weathersit"
        ],
        drop_first=True
    )

    # Make sure all 52 model features exist
    input_encoded = input_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Scale numerical features
    input_encoded[["temp", "hum", "windspeed"]] = scaler.transform(
        input_encoded[["temp", "hum", "windspeed"]]
    )

    # Prediction
    prediction = model.predict(input_encoded)

    st.success(
        "Predicted Bike Rental Demand: "
        + str(round(prediction[0]))
    )

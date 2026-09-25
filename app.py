import streamlit as st
import pandas as pd
import gzip
import pickle
from datetime import datetime


# -----------------------------------
# Page Title
# -----------------------------------

st.title("Bike Rental Demand Prediction")

st.write(
    "Enter the weather and time details to predict the expected number "
    "of bike rentals."
)


# -----------------------------------
# Load Model
# -----------------------------------

with gzip.open("model.pkl.gz", "rb") as f:
    model = pickle.load(f)


# -----------------------------------
# Load Scaler
# -----------------------------------

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


# -----------------------------------
# Load Feature Columns
# -----------------------------------

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)


# -----------------------------------
# User Inputs
# -----------------------------------

st.subheader("Enter Details")


# Season
season = st.selectbox(
    "Season",
    ["Spring", "Summer", "Fall", "Winter"]
)


# Time
hour_options = {
    "12:00 AM": 0,
    "1:00 AM": 1,
    "2:00 AM": 2,
    "3:00 AM": 3,
    "4:00 AM": 4,
    "5:00 AM": 5,
    "6:00 AM": 6,
    "7:00 AM": 7,
    "8:00 AM": 8,
    "9:00 AM": 9,
    "10:00 AM": 10,
    "11:00 AM": 11,
    "12:00 PM": 12,
    "1:00 PM": 13,
    "2:00 PM": 14,
    "3:00 PM": 15,
    "4:00 PM": 16,
    "5:00 PM": 17,
    "6:00 PM": 18,
    "7:00 PM": 19,
    "8:00 PM": 20,
    "9:00 PM": 21,
    "10:00 PM": 22,
    "11:00 PM": 23
}

selected_time = st.selectbox(
    "Time of Day",
    list(hour_options.keys())
)

hour = hour_options[selected_time]


# Holiday
holiday = st.selectbox(
    "Is it a Holiday?",
    ["No", "Yes"]
)


# Weekday
weekday_options = {
    "Sunday": 0,
    "Monday": 1,
    "Tuesday": 2,
    "Wednesday": 3,
    "Thursday": 4,
    "Friday": 5,
    "Saturday": 6
}

selected_weekday = st.selectbox(
    "Day of the Week",
    list(weekday_options.keys())
)

weekday = weekday_options[selected_weekday]


# Working Day
working_day = st.selectbox(
    "Is it a Working Day?",
    ["No", "Yes"]
)


# Weather
weather = st.selectbox(
    "Weather Situation",
    [
        "Clear",
        "Mist",
        "Light Rain/Snow",
        "Heavy Rain/Snow"
    ]
)


# Temperature
temp = st.number_input(
    "Temperature (normalized)",
    min_value=0.02,
    max_value=1.00,
    value=0.50,
    step=0.01
)


# Humidity
hum = st.number_input(
    "Humidity (normalized)",
    min_value=0.08,
    max_value=1.00,
    value=0.64,
    step=0.01
)


# Windspeed
windspeed = st.number_input(
    "Windspeed (normalized)",
    min_value=0.00,
    max_value=0.8507,
    value=0.19,
    step=0.01
)


# -----------------------------------
# Prediction Button
# -----------------------------------

if st.button("Predict Bike Rental Demand"):

    # -----------------------------------
    # Convert user inputs
    # -----------------------------------

    # Model uses:
    # 0 = 2011
    # 1 = 2012
    #
    # We use 2012 automatically.
    yr = 1

    # Use current month automatically
    mnth = datetime.now().month

    # Convert Yes/No to 0/1
    if holiday == "No":
        holiday_value = 0
    else:
        holiday_value = 1

    if working_day == "No":
        workingday_value = 0
    else:
        workingday_value = 1


    # -----------------------------------
    # Create Input DataFrame
    # -----------------------------------

    input_data = pd.DataFrame({
        "season": [season],
        "yr": [yr],
        "mnth": [mnth],
        "hr": [hour],
        "holiday": [holiday_value],
        "weekday": [weekday],
        "workingday": [workingday_value],
        "weathersit": [weather],
        "temp": [temp],
        "hum": [hum],
        "windspeed": [windspeed]
    })


    # -----------------------------------
    # Features used during training
    # -----------------------------------

    categorical_features = [
        "season",
        "yr",
        "mnth",
        "hr",
        "holiday",
        "weekday",
        "workingday",
        "weathersit"
    ]

    numerical_features = [
        "temp",
        "hum",
        "windspeed"
    ]


    # -----------------------------------
    # Set original categories
    # -----------------------------------

    input_data["season"] = pd.Categorical(
        input_data["season"],
        categories=[
            "Fall",
            "Spring",
            "Summer",
            "Winter"
        ]
    )

    input_data["yr"] = pd.Categorical(
        input_data["yr"],
        categories=[0, 1]
    )

    input_data["mnth"] = pd.Categorical(
        input_data["mnth"],
        categories=list(range(1, 13))
    )

    input_data["hr"] = pd.Categorical(
        input_data["hr"],
        categories=list(range(24))
    )

    input_data["holiday"] = pd.Categorical(
        input_data["holiday"],
        categories=[0, 1]
    )

    input_data["weekday"] = pd.Categorical(
        input_data["weekday"],
        categories=list(range(7))
    )

    input_data["workingday"] = pd.Categorical(
        input_data["workingday"],
        categories=[0, 1]
    )

    input_data["weathersit"] = pd.Categorical(
        input_data["weathersit"],
        categories=[
            "Clear",
            "Heavy Rain/Snow",
            "Light Rain/Snow",
            "Mist"
        ]
    )


    # -----------------------------------
    # One-Hot Encoding
    # -----------------------------------

    input_encoded = pd.get_dummies(
        input_data,
        columns=categorical_features,
        drop_first=True
    )


    # -----------------------------------
    # Match the 52 training features
    # -----------------------------------

    input_encoded = input_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )


    # -----------------------------------
    # Scale Numerical Features
    # -----------------------------------

    input_encoded[numerical_features] = scaler.transform(
        input_encoded[numerical_features]
    )


    # -----------------------------------
    # Make Prediction
    # -----------------------------------

    prediction = model.predict(input_encoded)[0]


    # -----------------------------------
    # Display Result
    # -----------------------------------

    st.success(
        f"Predicted Bike Rental Demand: {round(prediction)} bikes"
    )

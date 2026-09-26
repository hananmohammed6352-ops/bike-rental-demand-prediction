import streamlit as st
import pandas as pd
import gzip
import pickle
from datetime import datetime
import time


# -----------------------------------
# Page Configuration
# -----------------------------------

st.set_page_config(
    page_title="Bike Rental Prediction",
    page_icon="🚲",
    layout="wide"
)


# -----------------------------------
# Custom CSS
# -----------------------------------

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 30px;
}

.section-title {
    font-size: 24px;
    font-weight: 600;
    margin-top: 10px;
    margin-bottom: 15px;
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    border: 1px solid rgba(128,128,128,0.3);
    margin-top: 25px;
}

.result-number {
    font-size: 42px;
    font-weight: 700;
}

.info-box {
    padding: 15px;
    border-radius: 12px;
    border: 1px solid rgba(128,128,128,0.2);
    margin-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------------
# Title
# -----------------------------------

st.markdown(
    '<div class="main-title">🚲 Bike Rental Demand Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predict the expected number of bike rentals using machine learning'
    '</div>',
    unsafe_allow_html=True
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
# Input Section
# -----------------------------------

col1, col2 = st.columns(2)


# ===================================
# LEFT COLUMN
# ===================================

with col1:

    st.markdown(
        '<div class="section-title">🌦️ Weather Conditions</div>',
        unsafe_allow_html=True
    )

    season = st.selectbox(
        "Season",
        ["Spring", "Summer", "Fall", "Winter"]
    )

    weather = st.selectbox(
        "Weather Situation",
        [
            "Clear",
            "Mist",
            "Light Rain/Snow",
            "Heavy Rain/Snow"
        ]
    )

    temp = st.number_input(
        "Temperature (normalized)",
        min_value=0.02,
        max_value=1.00,
        value=0.50,
        step=0.01
    )

    hum = st.number_input(
        "Humidity (normalized)",
        min_value=0.08,
        max_value=1.00,
        value=0.64,
        step=0.01
    )

    windspeed = st.number_input(
        "Windspeed (normalized)",
        min_value=0.00,
        max_value=0.8507,
        value=0.19,
        step=0.01
    )


# ===================================
# RIGHT COLUMN
# ===================================

with col2:

    st.markdown(
        '<div class="section-title">🕒 Time & Calendar</div>',
        unsafe_allow_html=True
    )

    # Time options
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


    holiday = st.selectbox(
        "Is it a Holiday?",
        ["No", "Yes"]
    )


    working_day = st.selectbox(
        "Is it a Working Day?",
        ["No", "Yes"]
    )


# -----------------------------------
# Prediction Button
# -----------------------------------

st.markdown("---")

button_col1, button_col2, button_col3 = st.columns([1, 2, 1])

with button_col2:

    predict_button = st.button(
        "🚲 Predict Bike Rental Demand",
        use_container_width=True
    )


# -----------------------------------
# Prediction
# -----------------------------------

if predict_button:

    # Small loading animation
    with st.spinner("Analyzing conditions and predicting demand..."):
        time.sleep(1)


        # -----------------------------------
        # Convert Inputs
        # -----------------------------------

        # Automatically use 2012
        yr = 1

        # Automatically use current month
        mnth = datetime.now().month


        # Holiday
        if holiday == "No":
            holiday_value = 0
        else:
            holiday_value = 1


        # Working day
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
        # Feature Lists
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
        # Set Categories
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
        # Match Model Features
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
        # Prediction
        # -----------------------------------

        prediction = model.predict(input_encoded)[0]

        prediction = round(prediction)


    # -----------------------------------
    # Animated Result
    # -----------------------------------

    st.balloons()

    st.markdown(
        '<div class="result-box">',
        unsafe_allow_html=True
    )

    st.markdown(
        "### 🎯 Predicted Bike Rental Demand"
    )

    st.markdown(
        f'<div class="result-number">{prediction} 🚲</div>',
        unsafe_allow_html=True
    )

    st.write(
        "bikes are expected to be rented under these conditions."
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


    # -----------------------------------
    # Prediction Summary
    # -----------------------------------

    st.markdown("### 📋 Prediction Summary")

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    with summary_col1:
        st.metric("Season", season)

    with summary_col2:
        st.metric("Time", selected_time)

    with summary_col3:
        st.metric("Weather", weather)
